import jax 
import jax.numpy as jnp
import jax.random as rdm
import jnp.linalg as jnpla
from jax.typing import ArrayLike

def input(X:ArrayLike, Y:ArrayLike):
    # X is the n by p genotype matrix
    # Y is the n by k transformed pseudo-bulk gene expression matrix of a focal gene

    # Dimension of genotype matrix X
    n, p = X.shape

    # Dimension of pseudo-bulk gene expression matrix 
    n, k = Y.shape

def init_params():
    # Initialize parameters for scfm
    pass  

def compute_KL(gamma_l: ArrayLike, beta_l: ArrayLike):
    pass
        


def compute_ELBO(X:ArrayLike, Y:ArrayLike, encovar:ArrayLike, postselect:ArrayLike, postmean_l: ArrayLike, postmean_lj: ArrayLike, postcovar_lj):
    # Evaluate E_Q(B)
    E_1 = jnp.sum(jnpla.diag(postselect) @  postmean.T)
    # Evaluate E_Q(B^T X^T X B)
    X_j = X[ : j]
    E_2 = jnp.sum(jnp.sum(jnp.sum( postmean_lj * X_j.T @ X_j @ (postmean_lj @ postmean_lj^T + postcovar_lj))))
    # Evaluate the first term in ELBO
    first_term = -0.5 * (jnpla.trace(jnp.inv(encovar @ Y.T @ Y) -  2 * jnpla.trace(jnp.inv(encovar) @ X.T @ X @ E_1) + jnpla.trace(jnp.inv(encovar @ E_2))))
    - 0.5 * k * jnp.log(2*pi) - 0.5 * n  * jnp.log * jnpla.det(encovar)
    # Evaluate the second term in ELBO
    second_term = compute_KL()



def update_params():
    # update variational parameters by CAVI


    # 
    pass    