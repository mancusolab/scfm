from typing import NamedTuple

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


class _LResult(NamedTuple):
    resid: Array
    X: Array
    post: PosteriorParams
    prior: PriorParams


def _update_post(R_l: Array, X: Array, post: PosteriorParams, prior: PriorParams, l_index: int) -> PosteriorParams:
    n, k = R_l.shape

    XtX = jnp.sum(X * X, axis=0)

    # p, k, k
    prior_prec = jnp.inv(prior.var_b[l_index])
    post_prec = jnp.inv(prior.resid_var) * XtX[:, jnp.newaxis, jnp.newaxis] + prior_prec
    post_cov = jnp.inv(post_prec)
    post_mean = jnp.einsum("pkq,qk,kn,np->pk", post_cov, prior_prec, R_l, X)

    # Update the post.prob
    alpha = jax.nn.softmax(jnp.log(prior.prob) - stats.multivariate_normal.logpdf(post_mean, jnp.zeros(k), post_cov))

    # update the data structure
    post = PosteriorParams(
        prob=post.prob.at[l_index].set(alpha),
        mean_b=post.mean_b.at[l_index].set(post_mean),
        var_b=post.var_b.at[l_index].set(post_cov),
    )

    return post


def _update_prior(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> PriorParams:
    """
    :return:updated prior_resid_var
    """

    n, p = X.shape

    # Update prior for effect size covariance matrix
    tmp = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    prior_covar_b = jnp.einsum("lj,ljkq->lkq", post.prob, tmp)

    # Compute some statistics
    # We evaluated E_Q(B) to be a k by k matrix denoted B
    B = jnp.sum(post.prob * post.mean_b, axis=0)

    # Evaluate E_Q[B^T * X^T * X * B]
    # Evaluate E_Q[B^T B]
    pred = X @ B
    M = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    outer_moment = jnp.einsum("nj, nj, lj, ljkq->kq", X, X, post.prob, M) + pred.T @ pred

    # Update prior residual variance
    term1 = jnp.sum(Y * Y)  # tr(Y^T*Y)
    term2 = -2 * jnp.sum(Y * pred)  # -2tr(Y^T*X*E_Q(B))
    term3 = jnp.trace(outer_moment)  # tr(E_Q[B^T*X^T*X*B])

    resid_var = (term1 + term2 + term3) / n

    prior = PriorParams(
        resid_var,
        prior.prob,
        prior_covar_b,
    )

    return prior


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


def _fit_lth_effect(l_index: int, params: _LResult) -> _LResult:
    R, X, post, prior = params

    # TODO: do the CAVI updates for the lth effect
    R_l = R + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    # update posterior paramters for lth effect and jth SNP
    post = _update_post(R_l, X, post, prior, l_index)

    # update residual for next \ell effect
    R = R_l - X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    return _LResult(R, X, post, prior)


def _fit_model(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams
) -> tuple[Array, PosteriorParams, PriorParams]:
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
