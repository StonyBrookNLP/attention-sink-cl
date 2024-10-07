## Does RoBERTa Perform Better than BERT in Continual Learning: An Attention Sink Perspective
This is the repository for the COLM 24 paper: [Does RoBERTa Perform Better than BERT in Continual Learning: An Attention Sink Perspective](https://openreview.net/pdf/871a164cc29448a6b9d44abc428bcd4529d60d16.pdf).

### Abstract
This project studies the Continual learning (CL) problem with pre-trained language models. 
* Pe-trained models may allocate high attention scores to some 'sink' tokens, such as *[SEP]* tokens, which are ubiquitous across various tasks. Such attention sinks may lead to models' over-smoothing in single-task learning and interference in sequential tasks’ learning, downgrading models' CL capacities.
* We propose a **pre-scaling mechanism** that encourages attention diversity across all tokens. Specifically, it first scales the task's attention to the non-sink tokens in a probing stage, and then fine-tunes the model with scaling.
* Experiments show that pre-scaling yields substantial improvements in CL without experience replay, or progressively storing parameters from previous tasks.  

### Package Requirement
```
numpy == 1.16.2
torch == 1.9.1
transformers == 3.0.0
```

### Data
Each task data are stored in the ```./data``` directory. For tasks in GLUE benchmark, please download the data in [this link](https://gluebenchmark.com/tasks) to the corresponding sub directories. Other data can be downloaded [here](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ).

To preprocess the data, please run ```python preprocess.py``` in each sub directory. For Yahoo and DBPedia which need to be split to sub-class tasks, please run ```python preprocess_split.py```.

### Run
The scripts for running BERT and RoBERTa with pre-scaling is in ```./bert``` and ```./roberta``` directory, respectively. 

For BERT models, we show sample commands for prescaling in ```./bert/script_prescale.sh``` and for probing-and-then-fine-tuning in ```./bert/script_prescale.sh```. And for RoBERTa models we show similar ```.sh``` files.

### Citation
```
@inproceedings{
bai2024does,
title={Does Ro{BERT}a Perform Better than {BERT} in Continual Learning: An Attention Sink Perspective},
author={Xueying Bai and Yifan Sun and Niranjan Balasubramanian},
booktitle={First Conference on Language Modeling},
year={2024},
url={https://openreview.net/forum?id=VHhwhmtx3b}
}
```


