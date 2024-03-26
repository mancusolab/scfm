import jax.numpy as jnp
import jax.random as rdm


# Generate genotype matrix X and gene expression matrix Y
key = rdm.PRNGKey(0)
key, x_key = rdm.split(key)
key, y_key = rdm.split(key)
key, b_key = rdm.split(key)
N, p, k = 500, 100, 10
r2 = 0.5

beta = rdm.normal(b_key, shape=(p,))
X = rdm.normal(x_key, shape=(N, p))

g = X @ beta
s2g = jnp.var(g)
s2e = (1 - r2) / r2 * s2g

Y = g + jnp.sqrt(s2e) * rdm.normal(y_key, shape=(N,))
