import matplotlib.pyplot as plt
import numpy as np
import os

from mcmc.vanilla_sgld import vanilla_sgld
from mcmc.monge_sgld import monge_sgld
from mcmc.rmsprop_sgld import rmsprop_sgld

# Not using them
# from utils import kl_divergence
# from mcmc.shampoo_sgld import shampoo_sgld
# from mcmc.vanilla_sghmc import vanilla_sghmc

import models
from plotting_functions import plot_samples
import warnings

warnings.filterwarnings("ignore")

seed = 1
np.random.seed(seed)


def replace(dict1, dict2):
    dict3 = dict1.copy()
    for (key, value) in dict2.items():
        dict3[key] = value
    return dict3


M = models.funnel(sig2v=80)
L = 20000
x0 = np.ones(2)


def run_exp(data_fn, name, defaults, other_args, tuning_name=None, custom=False):
    print(name)
    samples = data_fn(**replace(defaults, other_args))
    if not os.path.exists("figs"):
        os.mkdir("figs")
    if tuning_name is not None:
        custom = False
        if not os.path.exists(f"figs/{tuning_name}"):
            os.mkdir(f"figs/{tuning_name}")
        file_name = f"figs/{tuning_name}/{name}.png"
        file_name_custom = None
    else:
        file_name = f"figs/{name}.png"
        file_name_custom = f"figs/{name}custom.png"

    plot_samples(
        samples,
        M,
        xlim=[-5.0, 9.0],
        ylim=[-25.0, 25.0],
        file_name=file_name,
        title=name,
        custom=custom,
        file_name_custom=file_name_custom,
    )


noise = 1.0


def noisy_grad_fn(x):
    dim = x.shape[0]
    noisy_grad = M.dlogp(x).reshape((dim, 1))
    noisy_grad += noise * np.random.randn(*noisy_grad.shape)
    return noisy_grad


# default hyper parameters
vanilla_sgld_defaults = dict(
    noisy_grad_fn=noisy_grad_fn,
    eta=0.3,
    L=L,
    x=x0,
    seed=seed,
)
monge_sgld_defaults = dict(
    noisy_grad_fn=noisy_grad_fn,
    eta=0.3,
    L=L,
    x=x0,
    alpha_2=1.0,
    seed=seed,
)
rmsprop_sgld_defaults = dict(
    noisy_grad_fn=noisy_grad_fn,
    eta=0.3,
    L=L,
    x=x0,
    seed=seed,
)
shampoo_sgld_defaults = dict(
    noisy_grad_fn=noisy_grad_fn,
    eta=0.3,
    L=L,
    x=x0,
    seed=seed,
)
# sghmc_defaults = dict(
#     noisy_grad_fn=noisy_grad_fn,
#     eta=0.3,
#     L=L,
#     x=x0,
#     lambd=0.7,
#     noise=1.0,
#     seed=seed,
# )

# Tune hyper parameters for VanillaSGLD
# for eta in [0.01, 0.0075, 0.005, 0.0025, 0.001]:
#     run_exp(
#         vanilla_sgld,
#         f"VanillaSGLD_{eta}",
#         vanilla_sgld_defaults,
#         dict(
#             L=2000000,
#             eta=eta,
#         ),
#         tuning_name="VanillaSGLD",
#     )

# Run VanillaSGLD here

run_exp(
    vanilla_sgld,
    "VanillaSGLD",
    vanilla_sgld_defaults,
    dict(
        L=2000000,
        eta=0.001,
    ),
    custom="2",
)

# Tune hyper parameters for MongeSGLD

# Run MongeSGLD here
run_exp(
    monge_sgld,
    "MongeSGLD",
    monge_sgld_defaults,
    dict(
        L=2000000,
        eta=0.003,
        alpha_2=0.1,
        lambd=0.7,
    ),
    custom="2",
)

# Tune hyper parameters for RMSpropSGLD
# for eta in [0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]:
#     # for eta in [0.25, 0.1, 0.075, 0.05]:
#     for lambd in [0.8, 0.9, 0.95, 0.99, 0.995, 0.999]:
#         run_exp(
#             rmsprop_sgld,
#             f"RMSpropSGLD_{eta}_{lambd}",
#             rmsprop_sgld_defaults,
#             dict(
#                 L=2000000,
#                 eta=eta,
#                 lambd=lambd,
#                 epsilon=0.0,
#             ),
#             tuning_name="RMSpropSGLD",
#         )

# Run RMSpropSGLD here
run_exp(
    rmsprop_sgld,
    "RMSpropSGLD",
    rmsprop_sgld_defaults,
    dict(
        L=2000000,
        eta=0.0025,
        lambd=0.995,
        epsilon=0.0,
    ),
    custom="2",
)

# Run ShampooSGLD here, HYPER PARAMS NOT TUNED
# run_exp(
#     shampoo_sgld,
#     "ShampooSGLD",
#     shampoo_sgld_defaults,
#     dict(
#         L=2000000,
#         eta=0.005,
#         lambd=0.995,
#         epsilon=0.0,
#     ),
# )
