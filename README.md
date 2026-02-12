# X-DSTF: Explainable Dual-Stream Transformer Fusion Framework for AI-generated
Image Classification
In this paper, we propose the X-DSTF framework, a dual-stream transformer-based architecture for synthetic media forensics. In the current release, we provide the complete model architecture, preprocessing pipeline, and efficiency benchmarking scripts. We also included a detailed Reproducibility Appendix specifying all hyperparameters, dataset splits, and experimental settings. 
<div align="center">
  <img src='./figures/pipeline.png' align="center" width=800>
</div>

## Environment Setup

You can install the required packages by running the command:
```bash
pip install -r requirements.txt
```
## Efficiency Benchmarking
To benchmark model complexity and runtime, you can independently choose to compute FLOPs, parameter count, or inference latency, depending on their evaluation needs. Each benchmarking component is modularized into a dedicated script to ensure clarity, reproducibility, and flexibility.

```bash
benchmarks/
├── compute_flops.py
├── compute_params.py 
├── compute_latency.py
```
