# Research Context & Agronomic Decision Framework

## Overview
SmartAgri AI serves as a prototype platform for research in **mobile-edge agricultural decision support systems (DSS)**. It tackles three primary challenges faced by smallholder farmers:
1. **Bandwidth Limitations**: Edge AI (TFLite) eliminates reliance on cloud APIs for initial diagnostic scans.
2. **Multi-Factor Health Assessment**: Moving beyond binary disease classification into a holistic 0–100 crop health score based on agronomic risk factors.
3. **Linguistic Accessibility**: Bilingual support (English and Urdu) tailored for South Asian agricultural contexts.

## Multi-Factor Health Score Equation
The composite crop health score $S \in [0, 100]$ is computed via weighted agronomic sub-indices:

$$S = 100 - \left( w_d \cdot D_{risk} + w_w \cdot W_{stress} + w_n \cdot N_{risk} + w_l \cdot (100 - L_{cond}) \right)$$

Where:
- $D_{risk} \in [0, 100]$: Disease severity risk factor (derived from CV detection confidence)
- $W_{stress} \in [0, 100]$: Water deficit / stress index (derived from soil moisture & temperature)
- $N_{risk} \in [0, 100]$: Nutrient deficiency risk factor
- $L_{cond} \in [0, 100]$: Visual leaf condition metric
- $w_d, w_w, w_n, w_l$: Weight factors ($\sum w = 1.0$)

## Disambiguation & Scientific Integrity
SmartAgri AI strictly demarcates:
- **On-Device Computer Vision**: Real edge ML model inference (or clearly tagged mock engine when weights are unpopulated).
- **Heuristic Agronomic Rules**: Expert-system logic for irrigation recommendations based on FAO Irrigation & Drainage guidelines.
