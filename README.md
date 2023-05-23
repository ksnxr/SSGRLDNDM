Properly documented code will be made openly available later.

# Sampling from funnel

Plot the results of sampling from funnel with identity metric, RMSprop metric and Monge metric. Gradient is corrupted by noise.

See directory **sgmcmc**.

Python version is `3.10.11`. The main dependencies are:

```
scipy==1.10.1
matplotlib==3.7.1
numpy==1.24.3
```

Run 

```
python two_d.py
```

in directory **sgmcmc**, and the resulting figures can be found in **sgmcmc/figs**.

# Neural network experiment

Our code is based on [bnn_priors](https://github.com/ratschlab/bnn_priors). Following their practice, we used [sacred](https://github.com/IDSIA/sacred) to record the experiments. The code for ShampooSGLD is partly based on [here](https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch). The code for computing the curvature is partly based on [here](https://github.com/kylematoba/lcnn/blob/main/estimate_curvature.py).

*Note that our code may not work (correctly) for scenarios not considered in our paper, e.g. different datasets, different thinning intervals, etc.*

Here we provide the raw scripts for running experiments.

Concerning the environment used to run the experiments, Python version is `3.9.13`.

Dependencies as recorded by sacred are

```
"bnn-priors==0.1.0",
"matplotlib==3.6.2",
"numpy==1.23.5",
"pyro-ppl==1.8.2",
"sacred==0.8.2",
"torch==1.13.0+cu116"
```

Here `bnn_priors` refers to our customized version. Some options they provide may not work in our codebase, and we have some custom options.

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

in current directory, where `()` denote optional.

## Evaluating running time

For evaluating the running time, install the `bnn_priors` version as provided in `bnn_time`.

Follow

```
python bnn_time/experiments/train_experiments.py --model model_name --data dataset_name --inference inference_method_name --lrs lrs --trials num_trials --prior prior_name --temperature 1.0 --sampling_decay flat --batch_size 100 --width width_for_mnist --save_samples False --cycles 1 --burnin_batches 1000 (--other_args other_args)
```

in current directory, where `()` denote optional.

## Viewing obtained results

The scripts that can be used to reproduce the results as reported in the paper are `plot_evaluations.ipynb`, `plot_experiments_results.ipynb` and `compare_time.ipynb` in directory `final_results`.

Note that the environment used to visualize the results is slightly different from the environment used to obtain the results.