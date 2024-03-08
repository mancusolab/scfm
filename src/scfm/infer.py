from typing import NamedTuple, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp

from jaxtyping import Array, ArrayLike


class PriorParams(eqx.Module):
    # likelihood params
    resid_var: Array

    # prior params
    prob: Array
    var_b: Array


class PosteriorParams(eqx.Module):
    # variational params
    mean_b: Array
    var_b: Array
    prob: Array


class _LResult(NamedTuple):
    resid: Array
    X: Array
    post: PosteriorParams
    prior: PriorParams


def _update_prior(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    pass


def _compute_elbo(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    pass


def _fit_lth_effect(l_index: int, params: _LResult) -> _LResult:
    R, X, post, prior = params

    R_l = R + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, :])
    # TODO: do the CAVI updates for the lth effect
    # update post
    R = R_l - X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, :])

    return _LResult(R, X, post, prior)


def _fit_model(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams
) -> Tuple[float, PosteriorParams, PriorParams]:
    L, p, k = post.mean_b.shape

    # update variational parameters
    R = Y - X @ jnp.sum(post.mean_b * post.prob, axis=0)
    init_params = _LResult(R, X, post, prior)
    _, _, post, prior = lax.fori_loop(0, L, _fit_lth_effect, init_params)

    # update prior (i.e. resid_var)
    prior = _update_prior(Y, X, post, prior)

    # compute ELBO
    elbo = _compute_elbo(Y, X, post, prior)

    return elbo, post, prior


def finemap(Y: ArrayLike, X: ArrayLike, L: int, tol: float = 1e-3, max_iter: int = 100):
    # todo: QC the input data; check dimensions match, etc.

    n, k = Y.shape
    n, p = X.shape

    # initialize model parameters
    prior = PriorParams(
        resid_var=jnp.ones(k),
        prob=jnp.ones(p) / p,
        var_b=jnp.array([jnp.eye(k) for _ in range(L)]),
    )
    post = PosteriorParams(
        mean_b=jnp.zeros((L, p, k)),
        var_b=prior.prior_var_b,
        prob=prior.prior_prob,
    )

    elbo = cur_elbo = -jnp.inf
    for train_iter in range(max_iter):
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        if jnp.fabs(cur_elbo < elbo) < tol:
            break

    # todo: combine into some object to return
    return
