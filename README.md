# MLLMRec: A Preference Reasoning Paradigm with Graph Refinement for Multimodal Recommendation

<div align=center>
<img src=".\images\MLLMRec_log.png" alt="log" width="80%" />
</div>


## Introduction

This repository provides the implementation of **MLLMRec**, a preference reasoning paradigm with graph refinement for multimodal recommendation.

This is the **updated version of the code corresponding to the revised experimental setting**. The manuscript and experimental results have also been updated accordingly in the latest version on [arXiv](https://arxiv.org/abs/2508.15304v3).

<img src=".\images\MLLMRec_framework.png" alt="framework" width="100%"  />

## Environment Requirement

- python 3.9
- Pytorch 2.5.1+cu124

## Dataset

The original dataset is the publicly available [Baby/Sports/Clothing](http://jmcauley.ucsd.edu/data/amazon/links.html) datasets from Amazon.

To help quickly test MLLMRec, we provide the processed Baby dataset the in the "`data/baby`" folder for directly running model training. The other two processed datasets have been uploaded to [Google Drive](https://drive.google.com/drive/folders/1-hmqynLgoQezYGotkGUuweUCF5FCQI45?usp=sharing). 

## How to run

1. **Reasoning Stage**

   Run the files in the "`reasoning`" folder. Please refer to "`reasoning/README.md`" for the specific operation process.

2. **model training**

   Enter the "`src`" folder and execute the following command:

   ```
   cd ./src
   python main.py -m MLLMRec -d baby
   ```

Other parameters can be set either through the command line or by using the configuration files located in "`src/configs/model/MLLMRec.yaml`" and "`src/configs/dataset/*.yaml`".

## Performance Comparison

<img src=".\images\MLLMRec_results.png" alt="framework" width="100%"  />
