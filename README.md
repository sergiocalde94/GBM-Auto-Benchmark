# GBM Auto Benchmark (Binary Classification Problems)

Gradient boosting machine algorithms that are fighting for Kaggle's throne, which one will win?

![Fight!](https://media.giphy.com/media/4IFOwmDGktcsM/source.gif)

# Installation

This project had been tested with python 3.6, so it's better to have an environment with this version of python to make it work.

As this package has `setup.py` aligned with `requiremetns.txt`, you only need to clone this repository and execute `pip install .` in the root of the project.

It's better if you have an isolated environment activated before doing pip install (conda, venv, pipenv...).

# How To Run?

Once you have installed all of the requirements, you run the app with the command `streamlit run app.py` and automatically **streamlit** will open your navigator (localhost, port 8501 by default) and the app is ready, you can play with it!

# Change the dataset?

You can use another dataset, you only have to set your configuration file like `config.ini` example.

- `path`: Location of the dataset, can be a path (relative or absolute) or an URL
- `sep`: Dataset separator
- `name`: Name of your dataset, useful for visualization purposes
- `target`: Target feature, the column you want to predict  
- `na_values`: Array with some different NA representations, like for example [UNKNOWN,NOT AVAILABLE]
- `test_size_proportion`: Test proportion to divide train and set when training the models
- `random_seed`: Random seed useful for reproducibility of different benchmarks


Note: This is an experiment, if you encounter with any issue, please let me know about it 
