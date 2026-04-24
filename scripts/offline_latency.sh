#!/usr/bin/env bash
set -euo pipefail

# =========================
# Usage:
# bash offline_latency.sh <model>
#
# Example:
# bash offline_latency.sh Qwen/Qwen2.5-VL-7B-Instruct
# =========================

MODEL=${1:-}

if [ -z "$MODEL" ]; then
    echo "Usage: bash offline_latency.sh <model>"
    exit 1
fi

BATCH_SIZE=1
IMAGE_ROOT="/image"
ENTRY="entry_image.py"

OUTPUT_DIR="./results/offline_latency"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="${OUTPUT_DIR}/offline_latency_${TIMESTAMP}.csv"

echo "model,batch_size,image_length,latency_ms" > "${OUTPUT_CSV}"

echo "=========================================="
echo "Offline latency evaluation"
echo "Model      : ${MODEL}"
echo "Batch size : ${BATCH_SIZE}"
echo "Image root : ${IMAGE_ROOT}"
echo "Entry      : ${ENTRY}"
echo "Output CSV : ${OUTPUT_CSV}"
echo "=========================================="

for IMAGE_DIR in "${IMAGE_ROOT}"/*; do
    if [ ! -d "${IMAGE_DIR}" ]; then
        continue
    fi

    IMAGE_LENGTH=$(basename "${IMAGE_DIR}")

    echo ""
    echo "Running image length: ${IMAGE_LENGTH}"
    echo "Image dir: ${IMAGE_DIR}"

    LOG_FILE="${OUTPUT_DIR}/offline_latency_${IMAGE_LENGTH}_${TIMESTAMP}.log"

    python "${ENTRY}" \
        --model "${MODEL}" \
        --image_dir "${IMAGE_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        2>&1 | tee "${LOG_FILE}"

    # 假设 entry_image.py 输出里有类似：
    # latency_ms: 123.45
    LATENCY_MS=$(grep -E "latency_ms|Latency|latency" "${LOG_FILE}" \
        | tail -n 1 \
        | grep -oE "[0-9]+(\.[0-9]+)?" \
        | tail -n 1)

    if [ -z "${LATENCY_MS}" ]; then
        echo "[WARN] Cannot parse latency for ${IMAGE_LENGTH}, set as -1"
        LATENCY_MS="-1"
    fi

    echo "${MODEL},${BATCH_SIZE},${IMAGE_LENGTH},${LATENCY_MS}" >> "${OUTPUT_CSV}"

    echo "Done image length: ${IMAGE_LENGTH}, latency_ms=${LATENCY_MS}"
done

echo ""
echo "=========================================="
echo "All offline latency experiments finished."
echo "Result saved to:"
echo "${OUTPUT_CSV}"
echo "=========================================="