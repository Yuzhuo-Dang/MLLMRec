# Data preprocessing （We have provided the results on the Baby dataset）

## The preprocessing process of the raw data in the MMRec framework is from step 1 to step 4.

- The following preprocessing steps can be quite tedious. Please post issues if you cannot run the scripts.

- datasets: [Baby/Sports/Clothing](http://jmcauley.ucsd.edu/data/amazon/links.html) datasets from Amazon  
-- Rating file in `Files/Small subsets for experimentation`  
-- Meta files in `Per-category files`, [metadata], [image features]  

There has been an issue with the dataset site lately, 
as it automatically redirects to an updated version of the dataset. 
Keep pressing `ESC` to stop the redirecting action.

## Step by step

​	**step 1:** Performing 5-core filtering, re-indexing - `run 0rating2inter.ipynb`

​	**step 2:** Train/valid/test data splitting - `run 1spliting.ipynb`

​	**step 3:** Reindexing feature IDs with generated IDs in step 1 - `run 2reindex-feat.ipynb`

​	**step 4: **Encoding text/image features - `run 3feat-encoder.ipynb`



# Reasoning Strategy

## To facilitate the operation, we have placed the results we need after the above preprocessing in the "data" folder. Thus, you can skip the above four steps and start running directly from step 5.

Due to the file size restrictions of the submission system, we only provide the preprocessed results on the baby dataset (i.e., `baby.inter` and `meta-baby.csv` in the "`data/baby`" folder). The results for the other two datasets have been uploaded to Google Drive. If our work is accepted, we will make the links public on GitHub.

## Step by step

​	**step 5:** Semantic description generation - `run 4image2text.py`

​	**step 6:** User preference generation - `run 5preference-generation.py`

​	**step 7:** Encoding multimodal description - `run 6multimdoal-description-encoder.ipynb`

​	**step 8:** Encoding user preference - `run 7preference-encoder.ipynb`

To help quickly test MLLMRec, we also provide the processed results on the Baby dataset (i.e., `image_text_feat.npy` and `user_preferences_feat.npy` in the "`data/baby`" folder) for directly running model training.
