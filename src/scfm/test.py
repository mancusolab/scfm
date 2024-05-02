import jax
import jax.numpy as jnp
from scfm.infer import finemap
from jax import device_get
import pandas as pd
import jax.random as rdm
from jax.typing import ArrayLike
import jax
import warnings
import numpy as np
import jax.numpy.linalg as jnpla
from jaxtyping import Array, ArrayLike
import sys
#X_file = sys.argv[1]
#Y_file = sys.argv[2]
#X_df = pd.read_csv(X_file, sep="\t", header=None)
#Y_df = pd.read_csv(Y_file, sep="\t", header=None)
X_df = pd.read_csv("ENSG00000284688/ENSG00000284688.geno.tsv", sep="\t", header=None)
#Y_df = pd.read_csv("ENSG00000284688/ENSG00000284688_5_2_0.9_0.4_0.05_0.05.pheno.tsv", sep="\t", header=None)
    # Convert pandas DataFrames to NumPy arrays
#Y_np = Y_df.to_numpy()
X_np = X_df.to_numpy()
    # Convert NumPy arrays to JAX arrays
#Y = jnp.array(Y_np)
X = jnp.array(X_np)

# Sanity Check: Generate a geneotype matrix ~ binomial and simulate its phenotype Y with high heritablity and high LD and large sample size
def gen_X(N, p, prob):
    key = rdm.PRNGKey(0)
    X = rdm.binomial(key, n=2, p=prob, shape=(N, p))
    return X


def sim_pheno(X: ArrayLike, L: int, k: int, rho_g: float, rho_y: float, h2g: ArrayLike):
    key = rdm.PRNGKey(0)
    r_key, initial_key = rdm.split(key)
    N, p = X.shape

    # initialize B
    B = jnp.zeros((L, p, k))
    # h2g = jnp.full((k,), 0.01)
    I = jnp.eye(k)
    J = jnp.ones((k, k))
    S_G = h2g * I + rho_g * jnp.sqrt(jnp.outer(h2g, h2g)) * (J - I)
    # split keys for each L iteration
    keys = rdm.split(initial_key, num=L * 2)

    for l in range(L):
        b_key, g_key = keys[2 * l], keys[2 * l + 1]
        # For each l, compute B_l and store in B at index l
        Beta_l = rdm.normal(key=b_key, shape=(k,))
        indices = rdm.choice(g_key, p, shape=(1,), replace=False)
        print("indices that the selector choose:", indices)
        Gamma_l = jnp.zeros(p)
        Gamma_l = Gamma_l.at[indices].set(1)
        B_l = jnp.outer(Gamma_l, Beta_l)
        B = B.at[l, :, :].set(B_l)

    G = X @ B.sum(axis=0)
    C = jnp.cov(G.T)

    # Make sure C is positive definite
    epsilon = 0.001
    D = jnp.eye(k) * epsilon
    C = C + D

    # We want to scale C so that it matches S_G
    Lc = jnpla.cholesky(C)
    Lg = jnpla.cholesky(S_G)
    S = jnpla.solve(Lc.T, Lg.T)

    # Construct S_y
    S_y = (jnp.ones(k) - h2g) * I + (rho_y - rho_g) * jnp.sqrt(jnp.outer((jnp.ones(k) - h2g), (jnp.ones(k) - h2g))) * (
                J - I)
    # sim resid
    resid = rdm.multivariate_normal(r_key, jnp.zeros(k), S_y, shape=(N,))
    Y = G @ S + resid

    return Y, indices

#geno = gen_X(1000, 2000, 0.7)
her = jnp.array([0.9, 0.9])
pheno, index = sim_pheno(X, 1, 2, 0.3, 0.4, her)
#pheno_df = pd.DataFrame(np.array(pheno))
#pheno_df.to_csv("sim_pheno.csv", sep="\t", index=False)

prior, post, pip_all, pip_cs, cs, full_alphas, elbo, elbo_increase, l_order = finemap(pheno, X, 1)
#print(f"post = PosteriorParams(prob={device_get(post.prob)},\nmean_b={device_get(post.mean_b)},\nvar_b={device_get(post.var_b)})")
print(cs)
print(pip_cs[index])