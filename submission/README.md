# 50.007 Machine Learning Project

Before starting, do make sure that your project directory is in the following structure:

``` shell
data/
  EN/
    train
    dev.in
    dev.out
    test.in
  SG/
    train
    dev.in
    dev.out
  CN/
    train
    dev.in
    dev.out
models/
  hmm.py
  structured_perceptron.py
preprocess.py
main.py
```

Following which, you should be able to execute the commands below.

Generally the code can be run through the `main.py` file as follows:

``` shell
usage: main.py [-h] --part PART --datasets DATASETS [--epochs EPOCHS]

optional arguments:
  -h, --help           show this help message and exit
  --part PART          Possible parts: 2, 3, 4, 5, 5-laplace, 5-good_turing,
                       5-structured_perceptron
  --datasets DATASETS  Input datasets to be used, separated by commas.
                       Datasets should be stored in data/
  --epochs EPOCHS      Needed only when running 5-structured_perceptron or 5.
                       Defaults to 8.
```

Below are the instructions to run the specific parts of the project.

## Part 2

``` shell
python main.py --part=2 --datasets=EN,SG,CN
```

## Part 3

``` shell
python main.py --part=3 --datasets=EN,SG,CN
```

## Part 4

``` shell
python main.py --part=4 --datasets=EN
```

## Part 5

There are several models included in this part, HMM with laplace smooting, HMM with good turing estimate smoothing, and Structured Perceptron.

### HMM with Laplace Smoothing

``` shell
python main.py --part=5-laplace --datasets=EN
```

### HMM with Good Turning Estimate Smoothing

``` shell
python main.py --part=5-good_turing --datasets=EN
```

### Structured Perceptron

``` shell
python main.py --part=5-structured_perceptron --datasets=EN [--epochs=n]
```

`--epochs` is used to control the number of epochs to be trained before prediction on the dev set is made. We have set the default to 8 as per our results.

### Test Set

We have included a line which will allow predictions to be made on the test set too, as follows.

``` shell
python main.py --part=5 --datasets=EN [--epochs=n]
```

This runs the Structured Perceptron model and defaults to 8 epochs.
