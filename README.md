# Optimal Transport with Information Loss and Cluster Consistency for Domain Adaptation

**Course Project — EN.601.687: Machine Learning in Non-Stationary Environments**

**Authors**: Rishi More (`rmore2`), Chuan Lin (`clin146`)  
**Instructor**:  Anqi Liu

---

## 🔍 Overview

This project explores improved **domain adaptation** via **Optimal Transport (OT)** by incorporating:
- **Information Loss Regularization** to preserve label structure across domains.
- **Cluster Consistency Constraints** to align intra-class distributions.

We demonstrate superior performance over standard Sinkhorn OT on both synthetic and real-world benchmarks under **unsupervised** and **semi-supervised** settings.

---

## 🧠 Method Summary

We build on the Sinkhorn OT algorithm and introduce:

- **Info–Loss OT**: Minimizes KL divergence between source label distributions in the transport plan.
- **Cluster OT + Info–Loss**: Encourages intra-cluster consistency while preserving discriminative structure.

Our pipeline is evaluated using 1-NN classification on the transported source domain tested on the original target data.

---

## 📊 Results Summary

| Dataset               | No Adaptation | Sinkhorn OT | Cluster OT + Info–Loss | Info–Loss OT |
|-----------------------|---------------|-------------|--------------|-------------------------|
| Two–Moons             | 82.8%         | 87.2%       | 96.4%        | **99.8%**               |
| MNIST $\rightarrow$ USPS | 71.6%     | 77.4%       | 80.8%        | **82.7%**               |

See the `experiments` section in the notebook or `report.pdf` for full visualizations and t-SNE plots.
