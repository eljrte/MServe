#!/usr/bin/env bash
set -euo pipefail

# =========================
# Usage:
# bash offline_throughput.sh <model>
#
# Example:
# bash offline_throughput.sh Qwen/Qwen2.5-VL-7B-Instruct
# =========================

MODEL=${1:-}

if [ -z "${MODEL}" ]; then
    echo "Usage: bash offline_throughput.sh <model>"
    exit 1
fi

# =========================
# Paths
# =========================

ENTRY="entry.py"
TRACE_BUILDER="build_online_trace.sh"

TEXT_DATASET_PATH="./openchat_sharegpt4_dataset"
IMAGE_DATASET_PATH="./ShareGPT-4o"

TRACE_DIR="./traces/online"
OUTPUT_DIR="./results/offline_throughput"

mkdir -p "${TRACE_DIR}"
mkdir -p "${OUTPUT_DIR}"

# =========================
# Experiment Config
# =========================

REQ_RATES=(1 2 4 6 8 10 12 14 16 18 20)

NUM_REQUESTS=${NUM_REQUESTS:-1000}
SEED=${SEED:-42}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="${OUTPUT_DIR}/offline_throughput_${TIMESTAMP}.csv"

echo "model,req_per_s,trace_path,p99_ttft_ms,p99_tbt_ms,slo_attainment" > "${OUTPUT_CSV}"

echo "=========================================="
echo "Online serving throughput sweep"
echo "Model        : ${MODEL}"
echo "Entry        : ${ENTRY}"
echo "Trace builder: ${TRACE_BUILDER}"
echo "Req rates    : ${REQ_RATES[*]}"
echo "Num requests : ${NUM_REQUESTS}"
echo "Output CSV   : ${OUTPUT_CSV}"
echo "=========================================="

for RPS in "${REQ_RATES[@]}"; do
    echo ""
    echo "Generating trace for req/s=${RPS}"

    TRACE_PATH="${TRACE_DIR}/online_trace_rps${RPS}_${TIMESTAMP}.jsonl"

    NUM_REQUESTS="${NUM_REQUESTS}" \
    LAMBDA="${RPS}" \
    SEED="${SEED}" \
    bash "${TRACE_BUILDER}" \
        "${TEXT_DATASET_PATH}" \
        "${IMAGE_DATASET_PATH}" \
        "${TRACE_PATH}"

    echo "Running entry.py with trace: ${TRACE_PATH}"

    LOG_FILE="${OUTPUT_DIR}/offline_throughput_rps${RPS}_${TIMESTAMP}.log"

    python "${ENTRY}" \
        --model "${MODEL}" \
        --trace_path "${TRACE_PATH}" \
        --req_per_s "${RPS}" \
        2>&1 | tee "${LOG_FILE}"

    # =========================
    # Parse metrics
    # =========================

    P99_TTFT=$(grep -Ei "p99.*ttft|ttft.*p99" "${LOG_FILE}" \
        | tail -n 1 \
        | grep -oE "[0-9]+(\.[0-9]+)?" \
        | tail -n 1 || true)

    P99_TBT=$(grep -Ei "p99.*tbt|tbt.*p99" "${LOG_FILE}" \
        | tail -n 1 \
        | grep -oE "[0-9]+(\.[0-9]+)?" \
        | tail -n 1 || true)

    SLO_ATTAINMENT=$(grep -Ei "slo.*attainment|attainment.*slo" "${LOG_FILE}" \
        | tail -n 1 \
        | grep -oE "[0-9]+(\.[0-9]+)?" \
        | tail -n 1 || true)

    if [ -z "${P99_TTFT}" ]; then
        echo "[WARN] Cannot parse p99 TTFT for req/s=${RPS}"
        P99_TTFT="-1"
    fi

    if [ -z "${P99_TBT}" ]; then
        echo "[WARN] Cannot parse p99 TBT for req/s=${RPS}"
        P99_TBT="-1"
    fi

    if [ -z "${SLO_ATTAINMENT}" ]; then
        echo "[WARN] Cannot parse SLO attainment for req/s=${RPS}"
        SLO_ATTAINMENT="-1"
    fi

    echo "${MODEL},${RPS},${TRACE_PATH},${P99_TTFT},${P99_TBT},${SLO_ATTAINMENT}" >> "${OUTPUT_CSV}"

    echo "Done req/s=${RPS}: p99_ttft=${P99_TTFT}, p99_tbt=${P99_TBT}, slo_attainment=${SLO_ATTAINMENT}"
done

echo ""
echo "=========================================="
echo "All online serving throughput experiments finished."
echo "Result saved to:"
echo "${OUTPUT_CSV}"
echo "=========================================="