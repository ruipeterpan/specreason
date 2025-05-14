# SpecReason

This repo contains a proof-of-concept example of the system described in the paper SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning.


## Getting started

### Environment setup

- Create a conda environment: `conda create -n specreason python=3.12 -y`
- Activate the conda environment: `conda activate specreason`
- Install [vLLM 0.8.2](https://docs.vllm.ai/en/v0.8.2/getting_started/installation/gpu.html): `pip install vllm`
    - To run speculative decoding, you need to fix an issue in vLLM and install from source. See [spec_decoding_fix.md](spec_decoding_fix.md) for more details.
- Install other dependencies: `pip install datasets` (and whatever shows up in error messages 🙂)

### Launching vLLM servers

The script `spec_reason.py` requires two vLLM engines to be up and running. We use the following command to launch a 32B base model and a 1.5B small model on two A6000-48GB GPUs, both with TP=2.

You might need to adjust the ratio of how the KV cache memory space is partitioned (`--gpu-memory-utilization 0.1`) according to your hardware setup.

```shell
VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.8 --enable-prefix-caching --port 30000
VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.1 --enable-prefix-caching --port 30001
```

### Running SpecReason

```
mkdir results
OUTPUT_DIR=./results
python spec_reason.py --dataset_name aime --problem_id 60 --repeat_id 0 --score_threshold 7.0 --score_method greedy --token_budget 8192 --output_dir "$OUTPUT_DIR"
```

