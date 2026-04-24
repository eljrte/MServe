#!/usr/bin/env bash
set -euo pipefail

# =========================
# User Config
# =========================

TEXT_DATASET_PATH=${1:-"./openchat_sharegpt4_dataset"}
IMAGE_DATASET_PATH=${2:-"./ShareGPT-4o"}

OUTPUT_PATH=${3:-"./online_trace.jsonl"}

# 总请求数，必须为偶数，保证 text-only / text-image 各 50%
NUM_REQUESTS=${NUM_REQUESTS:-1000}

# Poisson process 的到达率 lambda，单位：requests / second
# 例如 LAMBDA=4 表示平均每秒 4 个请求
LAMBDA=${LAMBDA:-4.0}

# 随机种子
SEED=${SEED:-42}

python3 - <<PY
import os
import json
import random
import math
from pathlib import Path

TEXT_DATASET_PATH = Path("${TEXT_DATASET_PATH}")
IMAGE_DATASET_PATH = Path("${IMAGE_DATASET_PATH}")
OUTPUT_PATH = Path("${OUTPUT_PATH}")

NUM_REQUESTS = int("${NUM_REQUESTS}")
LAMBDA = float("${LAMBDA}")
SEED = int("${SEED}")

random.seed(SEED)

assert NUM_REQUESTS % 2 == 0, "NUM_REQUESTS must be even to keep 50% text-only and 50% text-image."
assert LAMBDA > 0, "LAMBDA must be positive."

# =========================
# Helpers
# =========================

def iter_json_objects(path: Path):
    """
    支持：
    1. 单个 .jsonl 文件
    2. 单个 .json 文件
    3. 一个目录，递归读取其中所有 .json / .jsonl
    """
    files = []

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("*.jsonl")) + list(path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    for fp in files:
        try:
            if fp.suffix == ".jsonl":
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            continue
            elif fp.suffix == ".json":
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for x in data:
                        yield x
                elif isinstance(data, dict):
                    # 常见格式：{"data": [...]}
                    for key in ["data", "conversations", "messages", "items"]:
                        if key in data and isinstance(data[key], list):
                            for x in data[key]:
                                yield x
                            break
                    else:
                        yield data
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")


def extract_text_from_conversations(obj):
    """
    兼容 ShareGPT / OpenChat 常见格式：
    - conversations: [{"from": "human", "value": "..."}]
    - messages: [{"role": "user", "content": "..."}]
    - instruction / input / output
    - prompt / query / question
    """
    if not isinstance(obj, dict):
        return None

    # 1. ShareGPT style
    if isinstance(obj.get("conversations"), list):
        parts = []
        for turn in obj["conversations"]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            value = turn.get("value", turn.get("content", ""))
            if isinstance(value, str) and value.strip():
                # 通常只取 user/human 的输入作为 prompt
                if role in ["human", "user", "instruction", ""]:
                    parts.append(value.strip())
        if parts:
            return "\n".join(parts)

    # 2. OpenAI messages style
    if isinstance(obj.get("messages"), list):
        parts = []
        for turn in obj["messages"]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            content = turn.get("content", "")
            if isinstance(content, str) and content.strip():
                if role in ["user", "human", ""]:
                    parts.append(content.strip())
            elif isinstance(content, list):
                # 多模态 content: [{"type":"text", "text":"..."}]
                for c in content:
                    if isinstance(c, dict):
                        text = c.get("text", "")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    # 3. instruction-tuning style
    candidates = []
    for key in ["prompt", "query", "question", "instruction", "input", "text"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    if candidates:
        return "\n".join(candidates)

    return None


def extract_image_path(obj):
    """
    尝试从常见字段中提取图像路径。
    如果没有找到，则返回 None。
    """
    if not isinstance(obj, dict):
        return None

    for key in ["image", "image_path", "image_file", "img", "img_path", "images"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and len(v) > 0:
            for item in v:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    for kk in ["path", "image", "file"]:
                        vv = item.get(kk)
                        if isinstance(vv, str) and vv.strip():
                            return vv.strip()

    # messages/content 中可能包含 image_url
    if isinstance(obj.get("messages"), list):
        for turn in obj["messages"]:
            content = turn.get("content", "") if isinstance(turn, dict) else ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        if "image_url" in c:
                            image_url = c["image_url"]
                            if isinstance(image_url, dict):
                                url = image_url.get("url")
                                if isinstance(url, str):
                                    return url
                            elif isinstance(image_url, str):
                                return image_url

    return None


def load_text_samples(path: Path):
    samples = []
    for obj in iter_json_objects(path):
        prompt = extract_text_from_conversations(obj)
        if prompt:
            samples.append({
                "prompt": prompt,
            })
    return samples


def load_image_samples(path: Path):
    samples = []
    for obj in iter_json_objects(path):
        prompt = extract_text_from_conversations(obj)
        image_path = extract_image_path(obj)

        # text-image 请求至少需要 prompt。
        # 如果 image_path 没解析出来，也保留，但 image 字段为 None。
        if prompt:
            samples.append({
                "prompt": prompt,
                "image": image_path,
            })
    return samples


def poisson_arrivals(n, lambd):
    """
    Poisson process:
    inter-arrival time ~ Exponential(lambda)
    """
    t = 0.0
    arrivals = []
    for _ in range(n):
        u = random.random()
        delta = -math.log(1.0 - u) / lambd
        t += delta
        arrivals.append(t)
    return arrivals


# =========================
# Load datasets
# =========================

text_samples = load_text_samples(TEXT_DATASET_PATH)
image_samples = load_image_samples(IMAGE_DATASET_PATH)

if len(text_samples) == 0:
    raise RuntimeError(f"No valid text samples loaded from {TEXT_DATASET_PATH}")

if len(image_samples) == 0:
    raise RuntimeError(f"No valid image samples loaded from {IMAGE_DATASET_PATH}")

num_text = NUM_REQUESTS // 2
num_image = NUM_REQUESTS // 2

# 如果数据不够，则有放回采样
text_chosen = [random.choice(text_samples) for _ in range(num_text)]
image_chosen = [random.choice(image_samples) for _ in range(num_image)]

requests = []

for x in text_chosen:
    requests.append({
        "modality": "text",
        "dataset": "openchat_sharegpt4_dataset",
        "prompt": x["prompt"],
    })

for x in image_chosen:
    requests.append({
        "modality": "text-image",
        "dataset": "ShareGPT-4o",
        "prompt": x["prompt"],
        "image": x.get("image"),
    })

# 混合请求顺序，使整体 50/50，但到达序列随机
random.shuffle(requests)

arrivals = poisson_arrivals(NUM_REQUESTS, LAMBDA)

for i, req in enumerate(requests):
    req["request_id"] = i
    req["arrival_time"] = arrivals[i]

# =========================
# Write output
# =========================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for req in requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

print("Done.")
print(f"Output: {OUTPUT_PATH}")
print(f"Total requests: {NUM_REQUESTS}")
print(f"Text-only requests: {num_text}")
print(f"Text-image requests: {num_image}")
print(f"Lambda: {LAMBDA} req/s")
print(f"Trace duration: {arrivals[-1]:.4f} s")
print(f"Text samples loaded: {len(text_samples)}")
print(f"Image samples loaded: {len(image_samples)}")
PY