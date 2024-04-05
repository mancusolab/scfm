import jax
import equinox as eqx
from jaxtyping import Array, ArrayLike

class PriorParams(eqx.Module):
    # residual covariance for Y
    resid_var: Array  # (k,k)

    # prior prob to select variable (p,)
    prob: Array  # (p,)

    # prior covariance for each effect over cell types
    var_b: Array  # (L,k,k)


class PosteriorParams(eqx.Module):
    # posterior prob to select variable (L,p)
    prob: Array  # (L,p)

    # posterior mean and covariance for each effect and variant
    mean_b: Array  # (L,p,k)
    var_b: Array  # (L,p,k,k)
