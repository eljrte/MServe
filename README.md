# MServe: An Efficient Serving System for Multimodal Large Language Model Reasoning

## Setup

### Hardware Requirements

**Server 1 (H100)**
- GPU: NVIDIA H100 (80GB VRAM)
- CPU: Intel Xeon Gold 6454S
- Memory: 512GB DDR5
- Interconnect: PCIe 5.0×16

**Server 2 (A30)**
- GPU: NVIDIA RTX A30 (24GB VRAM)
- CPU: Intel Xeon Gold 6330
- Memory: 256GB DDR4
- Interconnect: PCIe 4.0×16

### Software Environment

**A30 Environment**
- Ubuntu 24.04
- Python 3.10.8
- NVIDIA Driver 580.65.06
- CUDA 12.8
- PyTorch 2.7.1

**H100 Environment**
- Ubuntu 24.04
- Python 3.12.8
- NVIDIA Driver 580.65.06
- CUDA 12.8
- PyTorch 2.9.1

### Installation
```bash
# clone the repo
git clone https://github.com/eljrte/MServe

# Python dependencies
pip install -r requirements.txt
```

## Prepare models & datasets

### Models
- Qwen2.5-VL-32B
- Qwen2.5-VL-7B
- LLaVA-v1.6-34B
- LLaVA-v1.6-8B

### Datasets
- ShareGPT-4o
- MMDU
- openchat_sharegpt4_dataset

Models and datasets are publicly available on Hugging Face.

## Generate Online Trace

Traces used for online serving can be generated from the datasets using the following script:

```bash
chmod +x scripts/online_trace_generate.sh
cd scripts
NUM_REQUESTS=1000 LAMBDA=XX SEED=42 \
./online_trace_generate.sh \
./openchat_sharegpt4_dataset \
./ShareGPT-4o \
./online_trace.jsonl
```
Note: 'XX' is the number of requests per second.

## Run Offline Evaluation
```bash
cd scripts
./offline_latency.sh -M XX
./offline_throughput.sh -M XX
./offline_breakdown.sh -M XX
```
Note: 'XX' is the model name or path.

## Run Online Evaluation
```bash
cd scripts
./online_serving.sh
```


