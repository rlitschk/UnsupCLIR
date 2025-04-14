# Cross-Lingual Information Retrieval with Cross-Lingual Word Embeddings 

This repository contains the source code to reproduce the results for our experiments, which we presented at SIGIR'18 and SIGIR'19. 


## Installation

The following steps apply for both papers:
* Create an anaconda environment with `conda create -n unsupclir python=3.7`
* Install requirements: `pip install -r requirements.txt`
* Set up [clef-dataloaders](https://github.com/rlitschk/clef-dataloaders): `pip install git+https://github.com/rlitschk/clef-dataloaders.git`
* Download the CLEF 2000-2003 test collection from ELRA ([ELRA-E0008](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0008/))

# Reproduce SIGIR 2018 Results 
In our paper [Unsupervised Cross-Lingual Information Retrieval using Monolingual Data Only](https://arxiv.org/abs/1805.00879), we propose a fully unsupervised framework for ad-hoc cross-lingual information retrieval (CLIR) which requires no bilingual data at all. To reproduce our results, run `scripts/reproduce_sigir18.sh`. This will download cross-lingual word embeddings and run all experiments.

Bibtex:
```
@inproceedings{litschko2018unsupervised,
  title={Unsupervised cross-lingual information retrieval using monolingual data only},
  author={Litschko, Robert and Glava{\v{s}}, Goran and Ponzetto, Simone Paolo and Vuli{\'c}, Ivan},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={1253--1256},
  year={2018}
}
```

# Reproduce SIGIR 2019 Results 

In our paper [Evaluating Resource-Lean Cross-Lingual Embedding Models in Unsupervised Retrieval](https://dl.acm.org/doi/10.1145/3331184.3331324), we compare different CLWE spaces in CLIR. To reproduce the results, run `scripts/reproduce_sigir19_{clef,europarl}.sh`. 

Bibtex: 

```
@inproceedings{litschko2019evaluating,
  title={Evaluating resource-lean cross-lingual embedding models in unsupervised retrieval},
  author={Litschko, Robert and Glava{\v{s}}, Goran and Vulic, Ivan and Dietz, Laura},
  booktitle={Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval},
  pages={1109--1112},
  year={2019}
}
```