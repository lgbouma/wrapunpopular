#!/usr/bin/env bash
set -euo pipefail

# Launch run_many_ticids.py concurrently for each chunked target list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/scripts/targetlists"
LOG_DIR="${REPO_ROOT}/scripts/LOGS"

mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}"

for N in $(seq 1 16); do
    chunk_path="${TARGET_DIR}/20251103_scocen_quicklook_chunk${N}.csv"
    if [[ ! -f "${chunk_path}" ]]; then
        echo "Missing chunk file: ${chunk_path}" >&2
        exit 1
    fi

    echo "Launching chunk ${N}..."
    python -u run_many_ticids.py "${chunk_path}" &> "${LOG_DIR}/scocenql_${N}.txt" &
done

echo "All jobs launched. Check ${LOG_DIR} for logs."
