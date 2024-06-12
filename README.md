# RAG (Attributed Graph Refinement)
An official source code for paper **Attributed Graph Refinement via low rank
approximation and subspace learning**.

## Environments
The proposed RAG is implemented with python 3.8.8 on CPU.
No GPU required!
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
The 6 datasets we used: Cora, Citeseer, ACM, WiKi, DBLP, PubMed are all included in the ```data/``` directory.
All original attributed graph datasets is ```cora.mat```, ```citeseer.mat```, ```acm.mat```, ```dblp.mat```, ```pubmed.mat```. 

```cora_sorted.npz``` is the version after sorting by sample category, conveniently used to visualize matrix block diagonal effects.

```...Z1.npz``` is the result Z of completing the first stage of RAG for each dataset.
```...Z2.npz``` is the result Z of completing the second stage of RAG for each dataset.
These can used as the inputs in the multi-stage RAG ```RAGsSC.py```.

## Model
Our RAG model is in ```model.py```.


##Quick Start
Running node clustering of RAG: ```python RAGSC.py```.

Running node clustering of multi-stage RAG: ```python RAGsSC.py```.
