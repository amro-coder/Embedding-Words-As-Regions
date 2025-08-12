# Embedding Words As Regions: The Word2Ellipsoid Model

This repository contains the official source code for the Master's thesis, *"Embedding Words As Regions"*, completed at the University of Cambridge.

**[➡️ Read the Full Thesis Here](https://drive.google.com/file/d/1mloLsGVmMXoPhBCRZs0LAiwkUBsyrVYa/view?usp=sharing)**

---

## Abstract

> While Large Language Models (LLMs) have achieved state-of-the-art performance on a wide range of language understanding and generation benchmarks, their reliance on massive datasets and computational resources highlights a need for more data-efficient and representationally powerful models. A fundamental limitation, shared by both LLMs and traditional models like Word2Vec, is the use of point-based vectors, which struggle to capture complex, asymmetric linguistic relationships like hyponymy. Region-based representations offer a theoretically powerful alternative, modeling word meanings as regions where relationships like hyponymy can be represented by geometric containment.

> This thesis introduces Word2Ellipsoid, a novel region-based model that represents words as fuzzy hyperellipsoids derived from a Gaussian function. A key advantage of this approach is that, unlike existing box-based models, it provides a closed-form, approximation-free framework for calculating the volume of a hyperellipsoid. To evaluate its effectiveness, Word2Ellipsoid is benchmarked against a strong vector baseline (Word2Vec) and a state-of-the-art region baseline (Word2Box). All models were assessed on a comprehensive suite of word similarity and hyponymy detection tasks.

## Key Findings

The empirical findings from our experiments yield three principal conclusions:
1.  **Word2Ellipsoid outperformed Word2Box**, indicating that our approximation-free mathematical framework is more precise and robust.
2.  **The Word2Vec baseline consistently outperformed both regional methods**, suggesting it is a strong baseline whose capabilities may have been underestimated in prior research.
3.  **The evaluated regional models struggled to effectively capture hyponymy relations**, highlighting the inherent difficulty of learning such complex relationships from purely distributional data using simple training objectives.

---

## Getting Started

Follow these steps to set up the environment and reproduce the experiments.

### 1. Installation

First, clone the repository and install the required Python packages.

```bash
git clone https://github.com/your-username/Embedding-Words-As-Regions.git
cd Embedding-Words-As-Regions
pip install -r requirements.txt

### 2. Dataset Setup
The models were trained on the WaCkypedia_EN dataset.
Download: You can request access to the dataset from the official source:
https://wacky.sslmit.unibo.it/doku.php?id=start
Preprocess: The dataset is provided in XML format. The prepare_dataset.py script is provided to preprocess the data into the required format.

### 3. Configuration
Before running the code, you must configure the local paths.
Open the config.yaml file.
Edit the paths to point to the correct locations on your local machine where you have stored the dataset and where you want to save model outputs.
