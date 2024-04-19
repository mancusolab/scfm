import sys
import pytest
import jax.numpy as jnp
import jax.random as rdm
from .infer import finemap
import pandas as pd

def test_elbo_increase_in_finemap():
    X_df = pd.read_csv("ENSG00000012983.geno.tsv", sep="\t", header=None)
    Y_df = pd.read_csv("ENSG00000012983_3_2_0.3_0.4_0.05_0.05_2.pheno.tsv", sep="\t", header=None)
    # Convert pandas DataFrames to NumPy arrays
    Y_np = Y_df.to_numpy()
    X_np = X_df.to_numpy()
    # Convert NumPy arrays to JAX arrays
    Y = jnp.array(Y_np)
    X = jnp.array(X_np)
    iterations = 1
    elbo_increase = True
    previous_elbo = float('-inf')
    for _ in range(iterations):
        prior, post, pip_all, pip_cs, cs, full_alphas, elbo, elbo_increase, l_order = finemap(Y, X, 1)
        if elbo <= previous_elbo:
            elbo_increase = False
            break
        previous_elbo = elbo
    assert elbo_increase, "ELBO did not increase across iterations in finemap()"
