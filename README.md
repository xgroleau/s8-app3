# s8-app3

## Setup
You may need to create a virtual environment and download the dependencies. 
Make sure you have python3, python-venv and pip installed.

```sh
python3 -m venv .venv
#.venv/Scripts/activate # On windows
#source .venv/source bin/activat # On linux
pip install -r requirements.txt
```

## K Nearest Neighbors
To run the KNN classification script you can call

```sh
python -m src.classify_knn
``` 

In the script you can use `RELOAD_PARAMS` if you don't want to used the generated pickles.
If you want to graph the performance of KNN by the number of cluster you can set `PLOT_PERF_BY_N_REP` to `True`

## Bayes
To run the Bayes classification script you can call

```sh
python -m src.calssify_bayes
``` 