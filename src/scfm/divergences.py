import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla

from jax import vmap
from jax.scipy.special import xlogy
from jaxtyping import Array, ArrayLike

from src.scfm.infer import PosteriorParams, PriorParams


def _kl_mvn(m0: ArrayLike, sigma0: ArrayLike, sigma1: ArrayLike) -> float:
    """
    KL divergence between two multivariate normal distributions assuming prior mean is 0.
    """
    k, k = sigma1.shape

    factor = jspla.cho_factor(sigma1)
    term1 = jnp.trace(jspla.cho_solve(factor, sigma0)) - k

    rotated = jspla.solve_triangular(factor[0], m0)
    term2 = rotated.T @ rotated

    _, logdet1 = jnpla.slogdet(sigma1)
    _, logdet0 = jnpla.slogdet(sigma0)

    term3 = logdet1 - logdet0
    kl_mvn = 0.5 * (term1 + term2 + term3)

    return kl_mvn


# first vmap is to add dimensions for p variants
# second vmap is to add dimesions for L effects
_kl_mvn_vmap = vmap(vmap(_kl_mvn, (0, 0, None), 0), (0, 0, 0), 0)


def _kl_discrete(p: ArrayLike, q: ArrayLike) -> Array:
    """
    for the lth effect and jth SNP, evaluate KL distance between two multinomial distributions
    """
    return -jnp.sum(xlogy(p, q / p))


def kl_single_effect(prior: PriorParams, post: PosteriorParams) -> Array:
    """
    Sum the KL distance across all l and j for both MVN and Multinomial distributions

    Returns:
         The total KL distance
    """
    # weight each of the KL terms for effects by their respective alphas/probs
    kl_effects = jnp.sum(post.prob * _kl_mvn_vmap(post.mean_b, post.var_b, prior.var_b))
    kl_selections = _kl_discrete(post.prob, prior.prob)
    total_kl = kl_effects + kl_selections

    return total_kl
