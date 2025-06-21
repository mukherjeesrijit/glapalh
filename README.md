# **GLAPAL-H: Global, Local And Parts Aware Learner for Hydrocephalus Infection Diagnosis in Low-Field MRI**  
**IEEE Transactions of Biomedical Engineering 2025**
---

## 🎥 Tutorial

<p align="center">
  <a href="https://youtu.be/bvaHT2m4y3Q?si=lh9sL4PTJu2cShJq">
    <img src="https://img.youtube.com/vi/bvaHT2m4y3Q/hqdefault.jpg" alt="A 28 minutes' Tutorial for Domain-Experts">
  </a>
</p>

---

## 📚 References

1. (Free) Mukherjee, Srijit, et al. *"GLAPAL-H: Global, Local, And Parts Aware Learner for Hydrocephalus Infection Diagnosis in Low-Field MRI."*  medRxiv (2025): 2025-05. [📄 Read on medRxiv](https://www.medrxiv.org/content/10.1101/2025.05.14.25327461v2)
2. (IEEE) S. Mukherjee et al., "GLAPAL-H: Global, Local, And Parts Aware Learner for Hydrocephalus Infection Diagnosis in Low-Field MRI," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2025.3578541. [📄 Read on IEEE](https://ieeexplore.ieee.org/document/11029195)

---

## 🧠 Abstract

Hydrocephalus, marked by abnormal cerebrospinal fluid (CSF) accumulation, poses a global pediatric neurosurgical challenge, especially post-infectious hydrocephalus (PIH) in sub-Saharan Africa, which accounts for over 50% of cases, while non-post-infectious hydrocephalus (NPIH) stems from causes such ashemorrhage or congenital malformations; accurate differentiation among healthy, PIH, and NPIH infants is vital for effective management, as surgery may need to be deferred in active infections. While CT scans expose infants to ionizing radiation, low-field MRI offers a safer alternative, particularly in resource-constrained settings. However, the lower resolution of low-field MRI presents challenges for accurate diagnosis. 

To address the challenges of using low-field MRI for diagnosis, the study develops a custom approach that captures hydrocephalic etiology while simultaneously addressing quality issues encountered in low-field MRI. Specifically, we propose GLAPAL-H, a Global, Local, And Parts Aware Learner, which develops a multi-task architecture with global, local, and parts segmentation branches. The architecture segments images into brain tissue and CSF while using a shallow CNN for local feature extraction and develops a parallel deep CNN branch for global feature extraction. Three regularized training loss functions are developed — one for each of the global, local, and parts components. The global regularizer captures holistic features, the local focuses on fine details, and the parts regularizer learns soft segmentation masks that enable local features to capture hydrocephalic etiology. 

The study's results show that GLAPAL-H outperforms state-of-the-art alternatives, including CT-based approaches, for both Two-Class (PIH vs. NPIH) and Three-Class (PIH vs. NPIH vs. Healthy) classification tasks in accuracy, interpretability, and generalizability. GLAPAL-H highlights the potential of low-field MRI as a safer, low-cost alternative to CT imaging for pediatric hydrocephalus infection diagnosis and management. Practically, GLAPAL-H demonstrates robustness against the quantity and quality of  training imagery, enhancing its deployability. It further demonstrates computational efficiency, achieving reduced training times and low inference latency due to its etiology-guided learning design, making it suitable for rapid, scalable deployment in low-resource settings. We introduce the first domain-enriched AI approach for diagnosing infections in pediatric hydrocephalus using low-field MRI, a safer and 10 times more affordable alternative to CT scans, which pose radiation risks in children.---

## 🎯 Motivation

GLAPAL-H consists of three key branches:

- **Global Branch**: Captures holistic structural features via deep CNN.
- **Local Branch**: Focuses on fine-grained anatomical regions using shallow CNN.
- **Parts Branch**: Learns soft segmentation masks to align parts-level activations with hydrocephalic etiology.

Each component is trained with a specialized loss:
- Global Regularizer → holistic features  
- Local Regularizer → fine, localized cues  
- Parts Regularizer → soft attention on infection-relevant regions  

This design aligns with domain knowledge of disease manifestation, improving both **trust** and **performance**.

![Motivation](./GLAPALH_files/motivation.png)

---

## 🧩 GLAPAL-H: Model Overview


![GLAPAL-H Architecture](./GLAPALH_files/model.png)

---

## 📊 Results

- **Outperforms** state-of-the-art MRI and CT-based classifiers in both binary and ternary classification.
- **Improved generalizability** to new patients and imaging artifacts.
- **Faster training** and **lower inference time** due to shallow-deep hybrid design.

![Results](./GLAPALH_files/results.png)

---

## 🔍 Interpretability

- Activation maps reveal **clinically meaningful regions** associated with infection.
- Parts-aware attention boosts **explainability** and **clinical trust**.

![Activation Maps](./GLAPALH_files/activationmap.png)

---

