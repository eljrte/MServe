#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash offline_throughput.sh <model>

MODEL=${1:-}

if [ -z "$MODEL" ]; then
    echo "Usage: bash offline_throughput.sh <model>"
    exit 1
fi

IMAGE_ROOT="/image"
ENTRY="entry_batch.py"

BATCH_SIZES=(4 8 12 16 20)

OUTPUT_DIR="./results/offline_throughput"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="${OUTPUT_DIR}/offline_throughput_${TIMESTAMP}.csv"

echo "model,batch_size,latency_ms,throughput_req_per_s" > "${OUTPUT_CSV}"

echo "=========================================="
echo "Offline throughput evaluation"
echo "Model      : ${MODEL}"
echo "Image root : ${IMAGE_ROOT}"
echo "Entry      : ${ENTRY}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Output CSV : ${OUTPUT_CSV}"
echo "=========================================="

for BS in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "Running batch size: ${BS}"

    LOG_FILE="${OUTPUT_DIR}/offline_throughput_bs${BS}_${TIMESTAMP}.log"

    python "${ENTRY}" \
        --model "${MODEL}" \
        --image_dir "${IMAGE_ROOT}" \
        --batch_size "${BS}" \
        2>&1 | tee "${LOG_FILE}"

    # 从日志中解析 latency_ms
    # 兼容：
    # latency_ms: 123.45
    # Latency: 123.45 ms
    # latency: 123.45
    LATENCY_MS=$(grep -Ei "latency_ms|latency" "${LOG_FILE}" \
        | tail -n 1 \
        | grep -oE "[0-9]+(\.[0-9]+)?" \
        | tail -n 1 || true)

    if [ -z "${LATENCY_MS}" ]; then
        echo "[WARN] Cannot parse latency for batch size ${BS}, set latency=-1 throughput=-1"
        LATENCY_MS="-1"
        THROUGHPUT="-1"
    else
        THROUGHPUT=$(python3 - <<PY
bs = ${BS}
latency_ms = float("${LATENCY_MS}")
if latency_ms <= 0:
    print("-1")
else:
    print(f"{bs * 1000.0 / latency_ms:.4f}")
PY
)
    fi

    echo "${MODEL},${BS},${LATENCY_MS},${THROUGHPUT}" >> "${OUTPUT_CSV}"

    echo "Done batch size: ${BS}, latency_ms=${LATENCY_MS}, throughput=${THROUGHPUT} req/s"
done

echo ""
echo "=========================================="
echo "All offline throughput experiments finished."
echo "Result saved to:"
echo "${OUTPUT_CSV}"
echo "=========================================="