# Contextual Embedding Bias Measure
Code for the paper *Measuring Bias in Contextualized Word Representations* (to appear at GeBNLP 2019: 1st ACL Workshop on Gender Bias for Natural Language Processing)

See `notebooks` for analysis and `lib` for utility libraries.

## Requirements
- Python 3.6 or higher
- PyTorch 1.0 or higher

If you find our work useful, please consider citing the below paper:
```
@article{kurita2019measuring,
  title={Measuring Bias in Contextualized Word Representations},     
  author={Kurita, Keita and Vyas, Nidhi and Pareek, Ayush and Black, Alan W and Tsvetkov, Yulia},  
  journal={arXiv preprint arXiv:1906.07337},  
  year={2019}  
}
```
# This fork for contributes the following:
## Installation
1. From any of the notebooks, clone the Git repo (original or this fork):

`!git clone https://github.com/keitakurita/contextual_embedding_bias_measure.git`

2. Install the specific versions of allennlp and overrides:

`!pip install allennlp==0.9.0 spacy==2.1.4 overrides==3.1.0 -q`

3. Add the following code to download the spacy model (as a warmup):

```
#warmup of spacy
from allennlp.common.util import get_spacy_model
try:
  get_spacy_model("en", pos_tags=False, parse=True, ner=False)
except:
  pass
```

## Gender bias 
- [libs/bias_utils.py](libs/bias_utils.py) - bias calculation utilities grouped in a class: `BiasUtils`
- [notebooks/gender_bias_in_stress_detection_model.ipynb](notebooks/gender_bias_in_stress_detection_model.ipynb) - Gender bias experiments for stress detection

