import sys
import pytest
import jax.numpy as jnp
import jax.random as rdm
from .infer import finemap

def prepare_test_data(N, p, k):
    key = rdm.PRNGKey(0)
    key, x_key = rdm.split(key)
    key, y_key = rdm.split(key)
    key, b_key = rdm.split(key)
    r2 = 0.5

    beta = rdm.normal(b_key, shape=(p,k))
    X = rdm.normal(x_key, shape=(N, p))

    g = X @ beta
    s2g = jnp.var(g)
    s2e = (1 - r2) / r2 * s2g

    Y = g + jnp.sqrt(s2e) * rdm.normal(y_key, shape=(N,k))
    return Y, X

def test_elbo_increase_in_finemap(Y, X): # Ensure this function is defined correctly
    iterations = 10
    elbo_increase = True
    previous_elbo = float('-inf')
    for _ in range(iterations):
        elbo, prior, post = finemap(Y, X, 10)  # Ensure finemap is defined and works as expected
        if elbo <= previous_elbo:
            elbo_increase = False
            break
        previous_elbo = elbo  # This should be dedexnted to ensure it executes every loop iteration.
    assert elbo_increase, "ELBO did not increase across iterations in finemap()"

Y, X = prepare_test_data(3, 2, 3)
test_elbo_increase_in_finemap(Y, X)