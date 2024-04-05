import sys
import pytest
import jax.numpy as jnp
import jax.random as rdm
from .infer import finemap


def simulate_ind_effect(N, p, k, j, L):
    key = rdm.PRNGKey(0)
    X = rdm.randint(key, shape=(N, p), minval=0, maxval=3)  # simulate genotype matrix
    beta = rdm.normal(key, shape=(p,))
    gamma = jnp.zeros(k)
    gamma = gamma.at[j].set(1)  # at jth position is 1 everywhere else is 0

    # Initialize B with the shape L x p x k
    B = jnp.zeros((L, p, k))

    for l in range(L):
        # For each l, compute B_l and store in B at index l
        B_l = jnp.outer(beta, gamma)
        B = B.at[l, :, :].set(B_l)

    Y = X @ B.sum(axis=0)  # This sums B across its first dimension (L) before multiplication

    return Y, X

def simulate_shared_effect(N, p, k, c, L):
    key = rdm.PRNGKey(0)
    X = rdm.randint(key, shape=(N, p), minval=0, maxval=3)  # simulate genotype matrix
    beta = rdm.normal(key, shape=(p,))
    gamma = jnp.zeros(k)
    gamma = gamma.at[0:c].set(1)

    # Initialize B with the shape L x p x k
    B = jnp.zeros((L, p, k))

    for l in range(L):
        B_l = jnp.outer(beta, gamma)
        B = B.at[l, :, :].set(B_l)

    Y = X @ B.sum(axis=0)

    return Y, X

def simulate_correlated_effect(N, p, k, c, L_index):
    key = rdm.PRNGKey(0)
    X = rdm.randint(key, shape=(N, p), minval=0, maxval=3)  # simulate genotype matrix
    # Cholesky decomposition to transform standard normals into correlated normals
    rho = 0.9  # 90% correlation
    cov_matrix_2d = jnp.array([[1, rho], [rho, 1]])  # Covariance matrix for the first two betas
    L = jnp.linalg.cholesky(cov_matrix_2d)  # Cholesky decomposition

    # Generate standard normal variables
    key, subkey = rdm.split(key)
    z = rdm.normal(subkey, shape=(2,))

    # Transform using the Cholesky factor to get the correlated betas
    beta_2d = jnp.dot(L, z)

    # Generate the remaining p-2 betas independently
    key, subkey = rdm.split(key)
    beta_rest = rdm.normal(subkey, shape=(p - 2,))

    # Concatenate to form the complete beta vector
    beta = jnp.concatenate([beta_2d, beta_rest])

    gamma = jnp.zeros(k)
    gamma = gamma.at[0:c].set(1)

    # Initialize B with the shape L x p x k
    B = jnp.zeros((L_index, p, k))

    for i in range(L_index):
        B_l = jnp.outer(beta, gamma)
        B = B.at[i, :, :].set(B_l)

    Y = X @ B.sum(axis=0)  # assume 100% heritable
    return Y, X

@pytest.fixture
def test_data():
    N, p, k, c, L_index = 4, 3, 5, 3, 2
    Y, X = simulate_shared_effect(N, p, k, c, L_index)
    return Y, X


def test_elbo_increase_in_finemap(test_data):
    Y, X = test_data
    iterations = 1
    elbo_increase = True
    previous_elbo = float('-inf')
    for _ in range(iterations):
        elbo, prior, post = finemap(Y, X, 2)
        if elbo <= previous_elbo:
            elbo_increase = False
            break
        previous_elbo = elbo
    assert elbo_increase, "ELBO did not increase across iterations in finemap()"
