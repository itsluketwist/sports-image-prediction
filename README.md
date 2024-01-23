# **sports-image-prediction**

![check code workflow](https://github.com/itsluketwist/sports-image-prediction/actions/workflows/check.yaml/badge.svg)

A simple python project that takes an image of a sport, and uses a CNN to make a prediction about 
which sport it is!


<div>
    <!-- badges from : https://shields.io/ -->
    <!-- logos available : https://simpleicons.org/ -->
    <a href="https://opensource.org/licenses/MIT">
        <img alt="MIT License" src="https://img.shields.io/badge/Licence-MIT-yellow?style=for-the-badge&logo=docs&logoColor=white" />
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3" src="https://img.shields.io/badge/Python_3-blue?style=for-the-badge&logo=python&logoColor=white" />
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-red?style=for-the-badge&logo=pytorch&logoColor=white" />
    </a>
</div>


## *usage*

Follow `predict.ipynb` for easy access to the prediction code.

You can also run predictions (or re-train / re-evaluate the model) via the `src/run.py` interface.

Predictions via the command line, using `python`:

```shell
python src/run.py predict -i output/sports_mod_tuned.pth -s sample/basketball.pn
```

To retrain the model, check and update the hyperparameters in `src/train.py`, then run:

```shell
python src/run.py -m kmnist train
```


## *set-up and development*

Clone the repository code:

```shell
git clone https://github.com/itsluketwist/sports-image-prediction.git
```

Once cloned, install the requirements locally in a virtual environment:

```shell
python -m venv venv

. venv/bin/activate

pip install -r requirements-dev.txt
```

Install and use pre-commit to ensure code is in a good state:

```shell
pre-commit install

pre-commit autoupdate

pre-commit run --all-files
```


## *inspiration*

Wanted to learn more about neural networks, and get some experience building wqith PyTorch.
