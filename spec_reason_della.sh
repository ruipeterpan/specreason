#!/bin/bash
#SBATCH --job-name=specr_aime_32b_greedy9               # Job name
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=7:00:00                        # Time limit (24 hours max)
#SBATCH --mem=100G                            # Memory allocation (adjust as needed)
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.

CLUSTER="della"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "della" ]; then
    DATA_DIR="/scratch/gpfs/XXX/data"
    export HF_HOME="/scratch/gpfs/XXX/hf_cache"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    source /scratch/gpfs/XXX/miniconda3/etc/profile.d/conda.sh
    nvidia-smi
else
    exit 1
fi
conda activate specreason

# SpecR experiment configuration
DATASET_NAME="aime"
JUDGE_SCHEME="greedy"
THRESHOLD=9
NUM_REPEATS=16
BASE_MODEL_NAME="Qwen/QwQ-32B"  # "Qwen/QwQ-32B" or "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" or "NovaSky-AI/Sky-T1-32B-Preview"
SMALL_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL_ABBRV="Qwen-32B"  # "Qwen-32B" or "deepseek-70B" or "NovaSky-Preview"
SMALL_MODEL_ABBRV="deepseek-1.5B"  # "deepseek-1.5B" or "deepseek-7B"
OUTPUT_DIR="${DATA_DIR}/specreason/results/${JUDGE_SCHEME}_${THRESHOLD}/${DATASET_NAME}/${BASE_MODEL_ABBRV}_${SMALL_MODEL_ABBRV}"
LOGFILE_DIR="${DATA_DIR}/specreason/logs"
TP_SIZE=4  # applies for both models
TOKEN_BUDGET=8192
# Define the list of problem IDs
# ids1=($(seq 0 99))  # MATH and GPQA
ids1=($(seq 60 89))  # AIME


for dir in "$OUTPUT_DIR" "$LOGFILE_DIR"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    fi
done

# Function to check if a server is ready
wait_for_server() {
    local port=$1
    while true; do
        # Try to connect to the server
        curl -s http://localhost:$port/get_model_info > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done
}

# launch 32b model and 1.5b model one by one
vllm serve "$BASE_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 16384 --gpu-memory-utilization 0.75 --enable-prefix-caching --port 30000 &
VLLM_BASE_PID=$!
wait_for_server 30000
nvidia-smi
vllm serve "$SMALL_MODEL_NAME" --dtype auto -tp "$TP_SIZE" --max_model_len 16384 --gpu-memory-utilization 0.15 --enable-prefix-caching --port 30001 &
VLLM_SMALL_PID=$!
wait_for_server 30001


# Run for the first set of problem IDs
for id in "${ids1[@]}"; do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="${LOGFILE_DIR}/${timestamp}.log"
    echo "Running problem $id, parallelizing across the $NUM_REPEATS runs of a job, logfile $logfile"

    # Initialize an empty array to store process IDs
    pid_list=()

    for repeat_id in $(seq 0 $((NUM_REPEATS - 1))); do
        python spec_reason.py --dataset_name "$DATASET_NAME" --problem_id "$id" --repeat_id "$repeat_id" --score_threshold "${THRESHOLD}" --score_method "${JUDGE_SCHEME}" --token_budget "$TOKEN_BUDGET" --output_dir "$OUTPUT_DIR" >> "$logfile" 2>&1 &

        # Capture the PID of the last background process and store it in the list
        pid_list+=($!)
    done

    # Explicitly wait for only the processes in pid_list
    for pid in "${pid_list[@]}"; do
        wait "$pid"
    done

    # Print confirmation message
    echo "Problem $id completed"

    # Sleep for 2s
    sleep 2
done

kill $VLLM_BASE_PID
kill $VLLM_SMALL_PID
