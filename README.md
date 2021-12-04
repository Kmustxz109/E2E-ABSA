# E2E-ABSA
E2E-ABSA
# C-ATT-E2E-ABSA

## Requirements
* python 3.7.3
* pytorch 1.2.0
* transformers 2.0.0
* numpy 1.16.4
* tensorboardX 1.9
* tqdm 4.32.1
* some codes are borrowed from **allennlp** ([https://github.com/allenai/allennlp](https://github.com/allenai/allennlp), an awesome open-source NLP toolkit) and **transformers** ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers), formerly known as **pytorch-pretrained-bert** or **pytorch-transformers**)

## Architecture
* Pre-trained embedding layer: BERT-Base-Uncased (12-layer, 768-hidden, 12-heads, 110M parameters)

## Dataset
* ~~Restaurant: retaurant reviews from SemEval 2014 (task 4), SemEval 2015 (task 12) and SemEval 2016 (task 5) (rest_total)~~
* (**Important**) Restaurant: restaurant reviews from SemEval 2014 (rest14), restaurant reviews from SemEval 2015 (rest15), restaurant reviews from SemEval 2016 (rest16). Please refer to the newly updated files in ```./data```
* (**Important**) **DO NOT** use the ```rest_total``` dataset built by ourselves again, more details can be found in [Updated Results](https://github.com/lixin4ever/BERT-E2E-ABSA/blob/master/README.md#updated-results-important).
* Laptop: laptop reviews from SemEval 2014 (laptop14)


## Quick Start
* The valid tagging strategies/schemes (i.e., the ways representing text or entity span) in this project are **BIEOS** (also called **BIOES** or **BMES**), **BIO** (also called **IOB2**) and **OT** (also called **IO**). If you are not familiar with these terms, I strongly recommend you to read the following materials before running the program: 

  a. [Inside–outside–beginning (tagging)](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)). 
  
  
* Reproduce the results on Restaurant and Laptop dataset:
  ```
  # train the model with 5 different seed numbers
  python fast_run.py 
  ```
* Train the model on other ABSA dataset:
  
  1. place data files in the directory `./data/[YOUR_DATASET_NAME]` (please note that you need to re-organize your data files so that it can be directly adapted to this project, following the input format of `./data/laptop14/train.txt` should be OK).
  
  2. set `TASK_NAME` in `train.sh` as `[YOUR_DATASET_NAME]`.
  
  3. train the model:  `sh train.sh`

* (** **New feature** **) Perform pure inference/direct transfer over test/unseen data using the trained ABSA model:

  1. place data file in the directory `./data/[YOUR_EVAL_DATASET_NAME]`.
  
  2. set `TASK_NAME` in `work.sh` as `[YOUR_EVAL_DATASET_NAME]`
  
  3. set `ABSA_HOME` in `work.sh` as `[HOME_DIRECTORY_OF_YOUR_ABSA_MODEL]`
  
  4. run: `sh work.sh`


