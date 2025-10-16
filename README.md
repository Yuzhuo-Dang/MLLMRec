# MLLMRec: Exploring the Potential of Multimodal Large Language Models for Multimudal Recommendation



<div align=center>
<img src=".\images\MLLMRec_log.svg" alt="log" width="80%" />
</div>

## Introduction

This is the Pytorch implementation for our paper:

<img src=".\images\MLLMRec_framework.svg" alt="framework" width="100%"  />

## Environment Requirement

- python 3.9
- Pytorch 2.5.1+cu124

## Dataset

The original dataset is the publicly available [Baby/Sports/Clothing](http://jmcauley.ucsd.edu/data/amazon/links.html) datasets from Amazon.

To help quickly test MLLMRec, we provide the processed Baby dataset the in the "`data/baby`" folder for directly running model training. The other two processed datasets have been uploaded to Google Drive, and will be available upon acceptance. 

## How to run

1. **Reasoning Stage**

   Run the files in the "`reasoning`" folder. Please refer to `reasoning/README.md` for the specific operation process.

2. **model training**

   Enter the `src` folder and execute the following command:

   ```
   cd ./src
   python main.py -m MLLMRec -d baby
   ```

Other parameters can be set either through the command line or by using the configuration files located in `src/configs/model/MLLMRec.yaml` and `src/configs/dataset/*.yaml`.

## Performance Comparison

<img src=".\images\MLLMRec_results.svg" alt="framework" width="100%"  />
