
# Unsupervised Cross-Lingual Information Retrieval using Monolingual Data Only 

This project is the codebase for our paper "*Unsupervised Cross-Lingual Information Retrieval using Monolingual Data Only*" (UnsupCLIR) accepted at [SIGIR'18](http://sigir.org/sigir2018/). We propose a fully unsupervised framework for ad-hoc cross-lingual information retrieval (CLIR) which requires no bilingual data at all. Our experiments use the standard CLEF CLIR collections and outperform baselines that utilize cross-lingual embeddings relying on word- and document-level alignments.

Preprint: https://arxiv.org/abs/1805.00879


## Getting Started
In order to get started follow these steps:
* Get the official CLEF evaluation data from http://catalog.elra.info/product_info.php?products_id=888
    * Copy the data into the project folder to match the directory structure shown below (subfolder structure in /HOME/Data/)
* Download the embeddings from https://madata.bib.uni-mannheim.de/273/ and paste them into /HOME/Embeddings. Alternatively, you can also train your own shared embedding spaces.
	* Smith: https://github.com/Babylonpartners/fastText_multilingual
	* Conneau: https://github.com/facebookresearch/MUSE
* Set the HOME variable in constants<i></i>.py to point to the directory in which your project resides
* Run experiments_clef.py for individual results
* Run ensemble_clef.py to get ensemble results

Expected directory sturcture:

```bash
└── UnsupCLIR [HOME]
    ├── Data
    │   └── CLEF
    │       ├── DocumentData
    │       │   ├── dutch
    │       │   │   ├── algemeen_dagblad
    │       │   │   └── nrc_handelsblad
    │       │   ├── finnish
    │       │   │   └── aamu
    │       │   └── italian
    │       │       ├── la_stampa
    │       │       ├── sda_italian_94
    │       │       └── sda_italian_95
    │       ├── RelAssess
    │       │   ├── 2001
    │       │   ├── 2002
    │       │   └── 2003
    │       └── Topics
    │           ├── 2001
    │           ├── 2002
    │           └── 2003
    ├── Embeddings
    │   ├── Conneau
    │   │   ├── enfi
    │   │   ├── enit
    │   │   └── ennl
    │   ├── Smith
    │   │   ├── enfi
    │   │   ├── enit
    │   │   └── ennl
    │   └── Vulic
    │       ├── enfi
    │       ├── enit
    │       ├── ennl
    └── Results
```

## Reference

Reference to cite when you use UnsupCLIR in a research paper:

```
@inproceedings{LGPV18,
  title={Unsupervised Cross-Lingual Information Retrieval using Monolingual Data Only},
  author={Litschko, Robert and Glava\v{s}, Goran and Ponzetto, Simone Paolo and Vuli\'c, Ivan},
  booktitle={Proceedings of SIGIR},
  year={2018},
}
```

### Licence
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
