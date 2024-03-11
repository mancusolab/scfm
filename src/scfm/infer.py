from typing import NamedTuple, Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla

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


def data_loglikelihood(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    n, k = Y.shape
    n, p = X.shape
    D = jnp.sum(jnpla.diagonal(prior.prob) @ post.mean_b.T, axis=0)
    A = jnp.sum(prior.prob @ (post.mean_b @ post.mean_b.T + post.var_b), axis=(0, 1))
    ll = (
        -0.5
        * (
            jnp.trace(jnpla.inv(prior.resid_var) @ Y.T @ Y)
            - 2 * jnp.trace(jnpla.inv(prior.resid_var) @ Y.T @ X @ D)
            + jnp.trace(jnpla.inv(prior.resid_var)) @ X.T @ X @ A
        )
        - 0.5 * k * jnp.log(2 * jnp.pi)
        - 0.5 * n * jnp.log(jnpla.det(prior.resid_var))
    )
    return ll


def KL_MVN(m0: ArrayLike, sigma0: ArrayLike, m1: ArrayLike, sigma1: ArrayLike, k: int) -> float:
    term1 = jnp.trace(jnp.inv(sigma1) @ sigma0) - k
    term2 = (m1 - m0).T @ jnp.inv(sigma1) @ (m1 - m0)
    term3 = jnp.slogdet(sigma1) - jnp.slogdet(sigma0)
    KL = 0.5 * (term1 + term2 + term3)
    return KL


def _compute_elbo(ll: float, kl: float) -> float:
    cur_elbo = ll - kl
    return cur_elbo


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
    # check dimensions match
    n_y, k = Y.shape
    n_x, p = X.shape
    if n_y != n_x:
        raise ValueError("Number of individuals do not match: " f"Y is {n_y}x{k}, but X is {n_x}x{p}")

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

    ll = data_loglikelihood(Y, X, post, prior)
    kl = KL_MVN(Y, X, post, prior)
    cur_elbo = _compute_elbo(ll, kl)
    elbo = cur_elbo = -jnp.inf
    for train_iter in range(max_iter):
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        if jnp.fabs(cur_elbo < elbo) < tol:
            break

    # todo: combine into some object to return
    return
