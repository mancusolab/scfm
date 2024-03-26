from typing import NamedTuple, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
import jax.scipy.stats as stats

from jaxtyping import Array, ArrayLike

from .divergences import kl_single_effect


class PriorParams(eqx.Module):
    # residual covariance for Y
    resid_var: Array

    # prior prob to select variable (p,)
    prob: Array

    # prior mean and variance for each effect over cell types
    mean_b: Array
    var_b: Array


class PosteriorParams(eqx.Module):
    # posterior prob to select variable (L,p)
    prob: Array

    # posterior mean and covariance for each effect and variant
    mean_b: Array  # (L,p,k)
    var_b: Array  # (L,p,k,k)


class _LResult(NamedTuple):
    resid: Array
    X: Array
    post: PosteriorParams
    prior: PriorParams


def _update_prior(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams, l_index: int) -> PriorParams:
    M = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    # Update prior for effect size covariance matrix
    prior_update = jnp.einsum("lj, ljkq->kq", post.prob, M)
    prior.var_b = prior.var_b[l_index, :, :, :].set(prior_update)
    return prior


def _update_post(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams, l_index: int) -> PosteriorParams:
    p = X.shape(1)
    k = Y.shape(1)
    for j in range(p):
        # Update post.params using x = x.at[idx].set(y)
        # Update the post.var_b
        term1_var_b = jnp.inv(prior.resid_var) @ X[:, j].T @ X[:, j]
        term2_var_b = jnp.inv(prior.covar)
        post.var_b = post.var_b.at[l_index, j, :].set(term1_var_b + term2_var_b)

        # Update the post.mean_b
        R = Y - X @ jnp.sum(post.mean_b * post.prob, axis=0)
        R_l = R + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, :])
        post.mean_b = post.mean_b[l_index, j, :].set(
            post.var_b[l_index, j, :] @ jnp.inv(prior.resid_var) @ R_l.T @ X[:, j]
        )

        # Update the post.prob
        post.prob = post.prob[j].set(
            jax.nn.softmax(
                jnp.log(
                    prior.prob[:, j, :] - stats.normal.logpdf(jnp.zeros((k, 1))),
                    post.mean_b[l_index, j, :],
                    post.var_b[l_index, j, :, :],
                )
            )
        )
    return post


def _update_l(Y: Array, X: Array, prior: PriorParams, post: PosteriorParams, l_index: int):
    R = Y - X @ jnp.sum(post.mean_b * post.prob, axis=0)
    R_l = R + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])
    post_update = _update_post(Y, X, post, prior, l_index)
    prior_update = _update_prior(Y, X, post, prior, l_index)
    R = R_l - X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])
    return post_update, prior_update


def update_prior_resid_var(prior: PriorParams, post: PosteriorParams, X: Array, Y: Array):
    """
    :return:updated prior_resid_var
    """
    n = X.shape(0)
    # Compute some statistics
    # We evaluated E_Q(B) to be a k by k matrix denoted B
    B = jnp.sum(post.prob * post.mean_b, axis=0)

    # Evaluate E_Q[B^T * X^T * X * B]
    # Evaluate E_Q[B^T B]
    pred = X @ B
    M = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    outer_moment = jnp.einsum("nj, nj, lj, ljkq->kq", X, X, post.prob, M) + pred.T @ pred

    # Update prior residual variance
    term1 = jnp.trace(Y * Y.T)  # tr(Y^T*Y)
    term2 = -2 * jnp.trace(Y.T * pred.T)  # -2tr(Y^T*X*E_Q(B))
    term3 = jnp.trace(outer_moment)  # tr(E_Q[B^T*X^T*X*B])
    prior.resid_var = 1 / n * (term1 + term2 + term3)
    return prior.resid_var


def expected_loglikelihood(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    # Evaluate data log likelihood
    n, p = X.shape
    L, p, k = post.mean_b.shape

    # We evaluated E_Q(B) to be a k by k matrix denoted B
    B = jnp.sum(post.prob * post.mean_b, axis=0)

    # Evaluate E_Q[B^T * X^T * X * B]
    # Evaluate E_Q[B^T B]
    pred = X @ B
    M = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    outer_moment = jnp.einsum("nj, nj, lj, ljkq->kq", X, X, post.prob, M) + pred.T @ pred
    inv_prec_Yt = jspla.solve(prior.resid_var, Y.T, assume_a="pos")
    inv_prec_outer_moment = jspla.solve(prior.resid_var, outer_moment, assume_a="pos")

    ll = -0.5 * (
        (
            jnp.sum(inv_prec_Yt * Y.T)  # tr(inv(Sigma) Y'Y)
            - 2 * jnp.sum(inv_prec_Yt * pred.T)  # tr(inv(Sigma) Y'X E[B])
            + jnp.trace(inv_prec_outer_moment)  # tr(inv(Sigma) E[B'X'XB])
        )
        + k * jnp.log(2 * jnp.pi)
        + n * jnpla.slogdet(prior.resid_var)[1]
    )

    return ll


def _compute_elbo(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> Array:
    # compute log likelihood and kl distance
    ll = expected_loglikelihood(Y, X, post, prior)
    kl = kl_single_effect(prior, post)

    # Given the current loglikelihood and kl distance, evaluate the current ELBO
    cur_elbo = ll - kl

    return cur_elbo


def _fit_lth_effect(Y: Array, l_index: int, params: _LResult) -> _LResult:
    R, X, post, prior = params

    resid = Y - X @ jnp.sum(post.mean_b * post.prob, axis=0)
    # TODO: do the CAVI updates for the lth effect
    init_l_result = _LResult(R=resid, X=X, prior=prior, post=post)
    l_dim = post.mean_b.shape(0)
    # update the prior and post params for lth effect
    update_post, update_prior = lax.fori_loop(0, l_dim, _update_l(Y, X, post, prior, l_index), init_l_result)

    update_l_result = _LResult(R=resid, X=X, prior=update_prior, post=update_post)

    return update_l_result


def _fit_model(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams
) -> Tuple[Array, PosteriorParams, PriorParams]:
    L, p, k = post.mean_b.shape

    # Compute residuals and update model
    R = Y - X @ jnp.sum(post.mean_b * post.prob, axis=0)
    init_params = _LResult(R, X, post, prior)
    _, _, post, prior = lax.fori_loop(0, L, _fit_lth_effect, init_params)

    # update prior (i.e. resid_var)
    prior = _update_prior(Y, X, post, prior)

    # compute ELBO
    elbo = _compute_elbo(Y, X, post, prior)

    return elbo, post, prior


def finemap(Y: ArrayLike, X: ArrayLike, L: int, prior_var: float = 1e-3, tol: float = 1e-3, max_iter: int = 100):
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
        mean=jnp.zeros((L, p, k)),
        var_b=jnp.tile(prior_var * jnp.eye(k), (L, 1, 1)),
    )
    post = PosteriorParams(
        mean_b=jnp.zeros((L, p, k)),
        var_b=prior.prior_var_b,
        prob=jnp.tile(prior.prior_prob, (L, 1)),
    )

    # evaluate current elbo
    elbo = cur_elbo = -jnp.inf
    for train_iter in range(max_iter):
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        if jnp.fabs(cur_elbo < elbo) < tol:
            print(f"ELBO has converged. ELBO at the last iteration: {cur_elbo}")
            break

    # todo: combine into some object to return
    return
