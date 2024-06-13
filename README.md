# RAG (Attributed Graph Refinement)
An official source code for paper **Attributed Graph Refinement via low rank
approximation and subspace learning**.

## Environments
The proposed RAG is implemented with python 3.8.8 on CPU.
All results in the paper are from running on an i7-10700 CPU.

No GPU required!
Use of the GPU may cause slowdowns.

### Packages
+ numpy==1.22.4
+ scipy==1.6.2
+ sklearn==0.24.1
+ matplotlib==3.3.4

```requirements.txt``` contains versions of all packages in our environment. 
You can install the same environment using the following command:
```pip install -r requirements.txt```

If you are using Anaconda, an identical environment can also be created by using the following command:
```conda env create -f environment.yml```


## Datasets
The 6 datasets we used: Cora, Citeseer, ACM, WiKi, DBLP, PubMed.
The ```data/``` holds several small datasets that can be used as demos. 
The full dataset can be accessed at https://drive.google.com/drive/folders/10Y2uqmQy21HPfgKBvxMov1svskxkOxXf?usp=sharing .
If you want to run the full dataset, just download all the data and put them in the ```data/``` directory.

All original attributed graph datasets is ```cora.mat```, ```citeseer.mat```, ```acm.mat```, ```dblp.mat```, ```pubmed.mat```. 

```cora_sorted.npz``` and ```citeseer_sorted.npz``` is the version after sorting by sample category, conveniently used to visualize matrix block diagonal effects.

```...Z1.npz``` is the result Z of completing the first stage of RAG for each dataset.
```...Z2.npz``` is the result Z of completing the second stage of RAG for each dataset.
These can used as the inputs in the multi-stage RAG ```RAGsSC.py```.

## Model
Our RAG model is in ```model.py```.

<div align="center">
<img src="https://github.com/LiminLi-xjtu/RAG_model/blob/master/github-images/multi-stageRAG.png" width="75%" height="75%" />
<br>
Figure 1: The process of multi-stage RAG
</div>

## Quick Start
Running node clustering of RAG: ```python RAGSC.py```.

Running node clustering of multi-stage RAG: ```python RAGsSC.py```.
(Before running ```python RAGsSC.py```, please make sure that the full dataset has been downloaded in Google Drive and saved in the ```data/``` directory.)



