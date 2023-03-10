# Funnel experiment

See directory **sgmcmc**.

Packages that we used are listed in *environment.yml*.

Run 

```
python two_d.py
```

in directory **sgmcmc**.

# Neural network experiment

Our code is based on [bnn_priors](https://github.com/ratschlab/bnn_priors). Following their practice, we used [sacred](https://github.com/IDSIA/sacred) to record the experiments.

Note that we carried out the neural network experiments on a compute cluster, and some of our scripts are related to the cluster. Here we provide the raw scripts for running experiments.

Python version is `3.9.13`.

Dependencies as recorded by sacred are

```
"bnn-priors==0.1.0",
"matplotlib==3.6.2",
"numpy==1.23.5",
"pyro-ppl==1.8.2",
"sacred==0.8.2",
"torch==1.13.0+cu116"
```

Note that `bnn_priors` refers to our customized version. Some options they provide may not work in our codebase, and we have some custom options.

Some options used in the below instructions are

* `model_name` is `classificationdensenet` for MNIST experiments and `googleresnet` for CIFAR10 Gaussian prior experiments or `correlatedgoogleresnet` for CIFAR10 correlated  Normal prior experiments,
* `data_name` is `mnist` for MNIST experiments and `cifar10` for CIFAR10 experiments,
* available `inference_method_name` are `VanillaSGLD` (identity metric), `WenzelSGLD` (Wenzel metric), `pSGLD` (RMSprop metric), `MongeSGLD` (Monge metric) and `ShampooSGLD` (Shampoo metric),
* `lrs` are learning rates, given as numbers, e.g. `0.1`,
* `num_trials` is number of repeated trials to run,
* `prior_name` is name of prior, e.g. `gaussian`, `horseshoe`, `convcorrnormal`
* `width_for_mnist` is width of the network for MNIST experiments and can be specified to an arbitrary number (e.g. 50) for CIFAR10,
* `other_args` are other arguments, specifically for `MongeSGLD` is `monge_alpha_2={number}`, where `number` is the value for ``\alpha^2``.

## Evaluating performance

For evaluating the performances of BNN, install the `bnn_priors` version as provided in `bnn_performance`.

Follow

```
python bnn_performance/experiments/train_experiments.py --model model_name --data dataset_name --inference inference_method_name --lrs lrs --trials num_trials --prior prior_name --temperature 1.0 --sampling_decay flat --batch_size 100 --width width_for_mnist --save_samples False --cycles 20 --burnin_batches 1000 (--other_args other_args)
```

in current directory, where `()` denotes optional.

## Evaluating running time

For evaluating the running time, install the `bnn_priors` version as provided in `bnn_time`.

Follow

```
python bnn_time/experiments/train_experiments.py --model model_name --data dataset_name --inference inference_method_name --lrs lrs --trials num_trials --prior prior_name --temperature 1.0 --sampling_decay flat --batch_size 100 --width width_for_mnist --save_samples False --cycles 1 --burnin_batches 1000 (--other_args other_args)
```

in current directory, where `()` denotes optional.

## Acknowledgements

* Code for Bayesian neural network is based on [bnn_priors](https://github.com/ratschlab/bnn_priors).
* Code for calculation of curvature of neural network is based on [lcnn](https://github.com/kylematoba/lcnn).
