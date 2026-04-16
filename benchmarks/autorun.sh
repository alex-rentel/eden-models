#!/usr/bin/env bash
set -euo pipefail

# ─── Automated MLX Benchmark Pipeline ────────────────────
# Serves each model, runs benchmarks, collects results, generates comparison.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${BENCH_PORT:-8081}"
VENV="${BENCH_VENV:-$HOME/Bonsai-demo/.venv}"
API_URL="http://localhost:${PORT}/v1/chat/completions"
SERVER_PID=""
START_TIME=$(date +%s)

# ─── Model Registry ─────────────────────────────────────
declare -A MODEL_REPO
MODEL_REPO[bonsai-8b]="prism-ml/Bonsai-8B-mlx-1bit"
MODEL_REPO[bonsai-4b]="prism-ml/Bonsai-4B-mlx-1bit"
MODEL_REPO[bonsai-1.7b]="prism-ml/Bonsai-1.7B-mlx-1bit"
MODEL_REPO[qwen-7b]="mlx-community/Qwen2.5-7B-Instruct-4bit"
MODEL_REPO[qwen-3b]="mlx-community/Qwen2.5-3B-Instruct-4bit"
MODEL_REPO[qwen-1.5b]="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_REPO[llama-3b]="mlx-community/Llama-3.2-3B-Instruct-4bit"
MODEL_REPO[llama-1b]="mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_REPO[phi-3.5]="mlx-community/Phi-3.5-mini-instruct-4bit"
MODEL_REPO[gemma-2b]="mlx-community/gemma-2-2b-it-4bit"

# Ordered list for default run order (smallest first)
ALL_MODELS=(bonsai-1.7b bonsai-4b bonsai-8b qwen-1.5b qwen-3b qwen-7b llama-1b llama-3b phi-3.5 gemma-2b)

# ─── Colors ──────────────────────────────────────────────
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# ─── Hardware Detection ──────────────────────────────────
detect_hardware() {
    local chip ram_bytes ram_gb
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed 's/apple //;s/ /-/g')
    ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    ram_gb=$((ram_bytes / 1073741824))
    echo "${chip}-${ram_gb}gb"
}

HARDWARE=$(detect_hardware)
RESULTS_DIR="${SCRIPT_DIR}/results/${HARDWARE}"

# ─── Cleanup on Exit ────────────────────────────────────
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}Cleaning up server (PID $SERVER_PID)...${RESET}"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

# ─── Helpers ─────────────────────────────────────────────
log() { echo -e "${BOLD}[autorun]${RESET} $*"; }
err() { echo -e "${RED}[autorun] ERROR:${RESET} $*" >&2; }

wait_for_server() {
    local model="$1" timeout=120 elapsed=0
    log "Waiting for server to load ${CYAN}${model}${RESET}..."
    while (( elapsed < timeout )); do
        if curl -sf "${API_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
            >/dev/null 2>&1; then
            log "${GREEN}Server ready${RESET} (${elapsed}s)"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    err "Server failed to start within ${timeout}s"
    return 1
}

kill_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Wait for port to free
    local waited=0
    while lsof -i ":${PORT}" >/dev/null 2>&1 && (( waited < 15 )); do
        sleep 1
        waited=$((waited + 1))
    done
}

usage() {
    echo -e "${BOLD}MLX Bonsai Benchmarks — Automated Pipeline${RESET}"
    echo ""
    echo "Usage: $0 [OPTIONS] [model1 model2 ...]"
    echo ""
    echo "Options:"
    echo "  --list           List available models"
    echo "  --venv PATH      Path to Python venv (default: \$BENCH_VENV or ~/Bonsai-demo/.venv)"
    echo "  --port PORT      Server port (default: \$BENCH_PORT or 8081)"
    echo "  --results-dir D  Results directory (default: results/<hardware>/)"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run all models"
    echo "  $0 bonsai-8b qwen-7b            # Run specific models"
    echo "  $0 --venv ~/my-venv bonsai-8b   # Custom venv"
    echo "  $0 --port 8082                  # Custom port"
}

list_models() {
    echo -e "${BOLD}Available models:${RESET}"
    for key in "${ALL_MODELS[@]}"; do
        echo -e "  ${CYAN}${key}${RESET}  →  ${MODEL_REPO[$key]}"
    done
}

# ─── Parse Arguments ─────────────────────────────────────
SELECTED_MODELS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list|-l)    list_models; exit 0 ;;
        --help|-h)    usage; exit 0 ;;
        --venv)       VENV="$2"; shift 2 ;;
        --port)       PORT="$2"; API_URL="http://localhost:${PORT}/v1/chat/completions"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        -*)           err "Unknown option: $1"; usage; exit 1 ;;
        *)
            if [[ -z "${MODEL_REPO[$1]+x}" ]]; then
                err "Unknown model: $1"; list_models; exit 1
            fi
            SELECTED_MODELS+=("$1"); shift ;;
    esac
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

# ─── Validate ────────────────────────────────────────────
if [[ ! -f "${VENV}/bin/activate" ]]; then
    err "Venv not found at ${VENV}. Set --venv or BENCH_VENV."
    exit 1
fi

if lsof -i ":${PORT}" >/dev/null 2>&1; then
    err "Port ${PORT} is already in use. Kill the existing process or use --port."
    exit 1
fi

# ─── Main Loop ───────────────────────────────────────────
declare -A SCORES
FAILED_MODELS=()

echo -e "\n${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  MLX Benchmark Pipeline${RESET}"
echo -e "${DIM}  Hardware: ${HARDWARE}${RESET}"
echo -e "${DIM}  Models:   ${SELECTED_MODELS[*]}${RESET}"
echo -e "${DIM}  Results:  ${RESULTS_DIR}${RESET}"
echo -e "${DIM}  Port:     ${PORT}${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}\n"

for model_key in "${SELECTED_MODELS[@]}"; do
    model_repo="${MODEL_REPO[$model_key]}"
    model_results="${RESULTS_DIR}/${model_key}"
    mkdir -p "$model_results"

    echo -e "\n${BOLD}──────────────────────────────────────────────${RESET}"
    log "Starting ${CYAN}${model_key}${RESET} (${model_repo})"
    echo -e "${BOLD}──────────────────────────────────────────────${RESET}"

    # Start server
    # shellcheck disable=SC1091
    (
        source "${VENV}/bin/activate"
        python -m mlx_lm server --model "$model_repo" --port "$PORT" \
            > "${model_results}/server.log" 2>&1
    ) &
    SERVER_PID=$!

    # Wait for ready
    if ! wait_for_server "$model_repo"; then
        err "Skipping ${model_key} — server failed to start"
        FAILED_MODELS+=("$model_key")
        kill_server
        continue
    fi

    # Run benchmarks
    log "Running benchmarks..."
    (
        source "${VENV}/bin/activate"
        cd "$SCRIPT_DIR"
        BONSAI_MODEL="$model_repo" BONSAI_API_URL="$API_URL" \
            python run_all.py 2>&1 | tee "${model_results}/run_output.txt"
    ) || {
        err "Benchmark run failed for ${model_key}"
        FAILED_MODELS+=("$model_key")
    }

    # Move CSVs to model subfolder
    find "${SCRIPT_DIR}/results" -maxdepth 1 -name 'bench_*_*.csv' -newer "${model_results}/run_output.txt" \
        -exec mv {} "${model_results}/" \; 2>/dev/null
    # Also grab any CSVs created in the last 10 minutes at top level
    find "${SCRIPT_DIR}/results" -maxdepth 1 -name 'bench_*_*.csv' -mmin -10 \
        -exec mv {} "${model_results}/" \; 2>/dev/null

    # Extract score from run output
    score=$(grep -oP 'Overall Score:\s+.*?(\d+/\d+)' "${model_results}/run_output.txt" 2>/dev/null | grep -oP '\d+/\d+' || echo "?/?")
    SCORES[$model_key]="$score"
    log "${GREEN}${model_key}: ${score}${RESET}"

    # Kill server
    kill_server
    log "Server stopped."
done

# ─── Generate Comparison ─────────────────────────────────
echo -e "\n${BOLD}═══════════════════════════════════════════════════════════${RESET}"
log "Generating comparison table..."
(
    source "${VENV}/bin/activate"
    cd "$SCRIPT_DIR"
    python compare_results.py --results-dir "$RESULTS_DIR" 2>&1 || python compare_results.py 2>&1 || true
)

# ─── Summary ─────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo -e "\n${BOLD}═══════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  PIPELINE COMPLETE — ${ELAPSED_MIN}m ${ELAPSED_SEC}s${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}\n"

echo -e "${BOLD}  Model Scores:${RESET}"
for model_key in "${SELECTED_MODELS[@]}"; do
    score="${SCORES[$model_key]:-skipped}"
    if [[ "$score" == "skipped" || "$score" == "?/?" ]]; then
        echo -e "    ${RED}${model_key}: ${score}${RESET}"
    else
        echo -e "    ${GREEN}${model_key}: ${score}${RESET}"
    fi
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo -e "\n  ${RED}Failed: ${FAILED_MODELS[*]}${RESET}"
fi

echo ""
