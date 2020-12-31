# BERTScore
### Installation
* Python version >= 3.6
* PyTorch version >= 1.0.0

Install from pypi with pip by 

```sh
pip install bert-score
```
Install latest unstable version from the master branch on Github by:
```
pip install git+https://github.com/Tiiiger/bert_score
```

Install it from the source by:
```sh
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .
```
and you may test your installation by:
```
python -m unittest discover
```
Check [original repo](https://github.com/Tiiiger/bert_score) for more details.

### Usage
Some modifications are made on `bert_score_cli/score.py`.

Run:
```sh
bert-score -rc ../ --lang en
```

