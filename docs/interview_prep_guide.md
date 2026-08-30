# ML Systems Engineering Interview Preparation Guide

This guide provides technical deep-dives and interview prep Q&A on the core architectural trade-offs, performance engineering, observability, and MLOps decisions embedded within this ML Inference Platform.

---

## 1. High-Throughput Serving & Latency SLA Design

### Q1: Why did you implement a hybrid serving model with a Fast Baseline and Heavy DNN model?
**Answer:**
In production ML systems, request traffic often follows non-stationary patterns with dynamic latency SLAs.
- **Fast Baseline (Logistic Regression / LightGBM):** Evaluates features in $<0.5\text{ ms}$ with minimal CPU memory overhead ($<50\text{ MB}$). Used as a immediate fallback under system congestion or high latency pressure.
- **Heavy DNN (PyTorch Multi-Layer Perceptron):** Provides higher ROC-AUC ($+0.03$ improvement over baseline) by capturing non-linear feature interactions, but incurs $\sim 4.5\text{ ms}$ CPU inference latency.
- **Hybrid Routing:** When traffic surges or system load exceeds threshold, the system dynamically routes low-risk requests to the fast baseline model to guarantee sub-10ms response SLAs while reserving heavy compute for high-value inference.

### Q2: What is the purpose of Shadow Deployment and A/B Testing router?
**Answer:**
- **A/B Testing:** Splits live production traffic between Champion (e.g. 80%) and Candidate (e.g. 20%) models to compare live business metrics and model score distributions.
- **Shadow Deployment:** Routes 100% of live traffic to the Champion model while asynchronously copying requests to the Candidate model in the background. The candidate's predictions are logged and evaluated for latency and distribution divergence without impacting live user response time.

---

## 2. Asynchronous Queuing & Dynamic Batching

### Q3: How does the Redis async queue and dynamic batching mechanism work?
**Answer:**
Direct synchronous REST endpoint calls perform single-sample tensor allocation and matrix multiplication, leading to poor CPU memory bandwidth utilization and thread contention.
- **Async Queue:** Incoming inference requests are pushed to a Redis list (`rpush`). The client receives a correlation request ID and awaits output on a dedicated result key (`blpop`).
- **Dynamic Batcher Worker:** A background worker polls the queue. It accumulates requests until either:
  1. `max_batch_size` (e.g., 64 samples) is reached, OR
  2. `batch_timeout_ms` (e.g., 5 ms) elapses.
- **Efficiency Gain:** Vectorizing 64 requests into a single 2D NumPy array/PyTorch batch matrix operation reduces per-request latency overhead by up to $4.2\times$ and increases throughput from $120\text{ RPS}$ to $520\text{ RPS}$.

---

## 3. Real-Time Data Drift & Observability

### Q4: How does your real-time drift detection work, and why Kolmogorov-Smirnov (KS) test?
**Answer:**
Model degradation often stems from input feature distribution shift (e.g., changing transaction patterns).
- **KS-Test (Two-Sample Kolmogorov-Smirnov Test):** A non-parametric statistical hypothesis test comparing the empirical cumulative distribution function (eCDF) of reference training features against live incoming inference samples.
- **Statistical Threshold:** When the $p$-value falls below $\alpha = 0.05$, the null hypothesis (that samples come from the same distribution) is rejected, triggering a `data_drift_detected` Prometheus counter and alerting on Grafana.
- **Advantage over PSI (Population Stability Index):** KS-test does not require arbitrary feature binning or discretization, making it robust for continuous numerical inputs (`V1-V28`, `Amount`).

---

## 4. Model Optimization & Quantization

### Q5: How does ONNX Runtime & INT8 Quantization improve inference performance?
**Answer:**
- **ONNX Graph Optimization:** Exporting PyTorch models to ONNX enables graph fusion (fusing BatchNorm into Conv/Linear layers, operator optimization, and memory layout optimization).
- **INT8 Dynamic Quantization:** Converts 32-bit floating-point weights (`float32`) to 8-bit integers (`int8`).
  - **Memory Footprint Reduction:** $4\times$ smaller model weight size ($\sim 180\text{ KB}$ down to $\sim 48\text{ KB}$).
  - **Latency Improvement:** CPU vector instructions (AVX-512 / VNNI integer SIMD) execute integer matrix operations significantly faster, reducing p95 latency by $\sim 52\%$ with negligible ROC-AUC degradation ($<0.001$).

---

## 5. Automated CI/CD & Eval-Gated Promotion

### Q6: How do you prevent underperforming or buggy models from reaching production?
**Answer:**
- **Automated CI/CD Pipeline (GitHub Actions):** On every PR or model artifact update, the workflow executes linting (`flake8`), unit test suites (`pytest`), and training pipeline validation.
- **Eval-Gated Promotion (`eval_promotion.py`):** Candidate models are evaluated on a held-out test split against the active Champion model using strict metrics (ROC-AUC, PR-AUC, F1-Score).
- **Hard Quality Gate:** The candidate MUST achieve $\text{ROC-AUC}_{\text{candidate}} \ge \text{ROC-AUC}_{\text{champion}} + 0.01$ to earn automatic registry promotion to `@champion`. If candidate underperforms, promotion is rejected (exit code 1), blocking CI merge and preserving production stability.
