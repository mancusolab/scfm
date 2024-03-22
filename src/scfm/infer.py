from typing import NamedTuple, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.stats as stats

from jaxtyping import Array, ArrayLike


class PriorParams(eqx.Module):
    # likelihood params
    resid_var: Array
    # resid_var: K x K symmetric matrix

    # prior params
    prob: Array
    # prob: L x p x 1 matrix

    # prior mean
    mean: Array
    # mean: L x p x k matrix of zeros

    var_b: Array
    # var_b: L x p x k x k matrix


class PosteriorParams(eqx.Module):
    # variational params
    mean_b: Array
    # mean_b: L x p x k x 1 matrix

    var_b: Array
    # var_b: L x p x k x k matrix

    prob: Array
    # prob: L x p x 1 vector


class _LResult(NamedTuple):
    resid: Array
    X: Array
    post: PosteriorParams
    prior: PriorParams


def _update_prior(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams, D: Array, M: Array, A: Array, l_index: int
) -> PriorParams:
    n = X.shape(0)

    # Update prior residual variance
    term1 = jnp.trace(Y.T @ Y)
    term2 = -2 * jnp.trace(jnp.inv(prior.resid_var) @ Y.T @ X @ D)
    term3 = jnp.trace(jnp.inv(prior.resid_var @ A))
    prior.resid_var = 1 / n * (term1 + term2 + term3)

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


def data_loglikelihood(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams, L: int) -> float:
    # Evaluate data log likelihood
    n, k = Y.shape
    n, p = X.shape

    # We evaluated E_Q(B) to be a k by k matrix denoted D
    diagonal = jnp.array([jnp.diag(post.prob[l_index, :]) for l_index in range(L)])
    D = jnp.einsum("lpp,lpk->pk", diagonal, post.mean_b)

    # We evaluated E_Q(B^T * X^T * X * B) to be a k by k matrix denoted A
    # We first evaluated mu_lj * mu_lj^T denoted as M
    M = jnp.einsum("lpk, lpq->lpkq", post.mean_b, post.mean_b) + post.var_b
    term1 = jnp.einsum("nj, nj, lj, ljkq->kq", X, X, post.prob, M)
    term2 = D.T @ X.T @ X @ D
    A = term1 + term2

    ll = (
        -0.5
        * (
            jnp.trace(jnpla.inv(prior.resid_var) @ Y.T @ Y)
            - 2 * jnp.trace(jnpla.inv(prior.resid_var) @ Y.T @ X @ D)
            + jnp.trace(jnpla.inv(prior.resid_var) @ A)
        )
        - 0.5 * k * jnp.log(2 * jnp.pi)
        - 0.5 * n * jnp.log(jnpla.det(prior.resid_var))
    )
    # return loglikelihood, E_Q(B), mu_lj * mu_lj^T, and E_Q(B^T * X^T * X * B) for update prior step
    return ll, D, M, A


def KL_MVN(m0: ArrayLike, sigma0: ArrayLike, m1: ArrayLike, sigma1: ArrayLike, k: int) -> float:
    """
    for lth effect and jth SNP, evaluate KL distance between two multivariate normal distributions
    """
    term1 = jnp.trace(jnpla.inv(sigma1) @ sigma0) - k
    term2 = (m1 - m0).T @ jnpla.inv(sigma1) @ (m1 - m0)
    sign1, logdet1 = jnpla.slogdet(sigma1)
    sign0, logdet0 = jnpla.slogdet(sigma0)
    term3 = sign1 * logdet1 - sign0 * logdet0
    kl_mvn = 0.5 * (term1 + term2 + term3)
    return kl_mvn


def KL_Multi(p: ArrayLike, q: ArrayLike) -> float:
    """
    for the lth effect and jth SNP, evaluate KL distance between two multinomial distributions
    """
    logterm = jnp.log(p) - jnp.log(q)
    kl_multi = jnp.sum(p * logterm, axis=0)
    return kl_multi


def KL(prior: PriorParams, post: PosteriorParams, L: int, p: int, k: int) -> float:
    """
    Sum the KL distance across all l and j for both MVN and Multinomial distributions

    Returns:
         The total KL distance
    """
    total_kl = 0
    for l_index in range(L):
        for j in range(p):
            kl_mvn = KL_MVN(
                post.mean_b[l_index, j, :],
                post.var_b[l_index, j, :, :],
                prior.mean_b[l_index, j, :],
                prior.var_b[l_index, j, :, :],
                k,
            )
            kl_multi = KL_Multi(post.prob[l_index, j], prior.prob[l_index, j])
            total_kl = kl_mvn + kl_multi

    return total_kl


def _compute_elbo(ll: float, kl: float) -> float:
    # Given the current loglikelihood and kl distance, evaluate the current ELBO
    cur_elbo = ll - kl
    return cur_elbo


def _fit_lth_effect(l_index: int, params: _LResult, p: int) -> _LResult:
    R, X, post, prior = params

    # R_l = R + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, :])
    # TODO: do the CAVI updates for the lth effect

    # update posterior paramters for lth effect and jth SNP

    # update prior parameters for the lth effect and jth SNP

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
        mean=jnp.zeros((L, p, k)),
        var_b=jnp.array([jnp.eye(k) for _ in range(L)]),
    )
    post = PosteriorParams(
        mean_b=jnp.zeros((L, p, k)),
        var_b=prior.prior_var_b,
        prob=prior.prior_prob,
    )

    # compute log likelihood and kl distance
    ll = data_loglikelihood(Y, X, post, prior, L)
    kl = KL(prior, post, L, p, k)

    # evaluate current elbo
    cur_elbo = _compute_elbo(ll, kl)
    elbo = cur_elbo = -jnp.inf

    for train_iter in range(max_iter):
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        if jnp.fabs(cur_elbo < elbo) < tol:
            print(f"ELBO has converged. ELBO at the last iteration: {cur_elbo}")
            break

    # todo: combine into some object to return
    return
