from typing import NamedTuple, Tuple

import pandas as pd

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.random as rdm
import jax.scipy.linalg as jspla
import jax.scipy.stats as stats

from jaxtyping import Array, ArrayLike

from . import log
from .divergences import kl_single_effect
from .params import PosteriorParams, PriorParams


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class _LResult(NamedTuple):
    resid: Array
    X: Array
    Y: Array
    post: PosteriorParams
    prior: PriorParams


class SCFMResult(NamedTuple):
    """
    priors: the final prior parameter for the inference
    posteriors: the final posterior parameter for the inference
    pip_all: the PIP for each SNP across L credible sets
    pip_cs: the PIP across credible sets that are not pruned
    cs: the credile sets output after filtering on purity
    alphas: the full credible sets before filtering on purity
    elbo: the final ELBO
    elbo_increase: A boolean to indicate whether ELBO increases during the optimizations
    l_order: the orginal order that scfm infers
    lfsr: the local false sign rate at each SNP for each cell type
    """

    prior: PriorParams
    post: PosteriorParams
    pip_all: Array
    pip_cs: Array
    cs: pd.DataFrame
    alphas: pd.DataFrame
    elbo: Array
    elbo_increase: bool
    l_order: Array
    lfsr: Array


def _update_lth_params(
    R_l: Array, X: Array, post: PosteriorParams, prior: PriorParams, l_index: int
) -> tuple[PriorParams, PosteriorParams]:
    """
    :return:updated prior_resid_var
    followed Zeyun's style
    """
    n, k = R_l.shape
    n, p = X.shape

    XtX = jnp.sum(X * X, axis=0)

    # k,k
    prior_prec = jnpla.inv(prior.var_b[l_index])
    resid_prec = jnpla.inv(prior.resid_var)

    # p,k,k
    post_cov = jnpla.inv(resid_prec * XtX[:, jnp.newaxis, jnp.newaxis] + prior_prec)
    rTXSigInv = (resid_prec @ R_l.T) @ X
    post_mean = jnp.einsum("pkq,qp->pk", post_cov, rTXSigInv)

    # p,
    alpha = jax.nn.softmax(jnp.log(prior.prob) - stats.multivariate_normal.logpdf(post_mean, jnp.zeros(k), post_cov))

    # Update prior for effect size covariance matrix
    post_mean_sq = jnp.einsum("pk,pq->pkq", post_mean, post_mean) + post_cov
    prior_covar_b = jnp.einsum("p,pkq->kq", alpha, post_mean_sq)

    prior = prior._replace(
        var_b=prior.var_b.at[l_index].set(prior_covar_b),
    )

    # update the data structure
    post = PosteriorParams(
        prob=post.prob.at[l_index].set(alpha),
        mean_b=post.mean_b.at[l_index, :, :].set(post_mean),
        var_b=post.var_b.at[l_index].set(post_cov),
    )

    return prior, post


def _update_resid_covar(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> PriorParams:
    n, p = X.shape

    # Compute some statistics
    # We evaluated E_Q(B) to be a k by k matrix denoted B
    B = jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)

    # Evaluate E_Q[B^T * X^T * X * B]
    # Evaluate E_Q[B^T B]
    pred = X @ B
    M = jnp.einsum("lpk,lpq,lp->pkq", post.mean_b, post.mean_b, post.prob) + jnp.einsum(
        "lp,lpkq->pkq", post.prob, post.var_b
    )
    outer_moment = jnp.einsum("nj,nj,jkq->kq", X, X, M) + pred.T @ pred

    # Update prior residual variance
    # term1 = jnp.sum(Y * Y)  # tr(Y^T*Y)
    term1 = Y.T @ Y
    # term2 = -2 * jnp.sum(Y * pred)  # -2tr(Y^T*X*E_Q(B))
    term2 = -2 * Y.T @ pred
    # term3 = jnp.trace(outer_moment)  # tr(E_Q[B^T*X^T*X*B])
    term3 = outer_moment

    resid_var = (term1 + term2 + term3) / n

    prior = prior._replace(
        resid_var=resid_var,
    )

    return prior


def expected_loglikelihood(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    # Evaluate data log likelihood
    n, p = X.shape
    L, p, k = post.mean_b.shape

    # We evaluated E_Q(B) to be a k by k matrix denoted B
    B = jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)

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
    resid, X, Y, post, prior = params

    # TODO: do the CAVI updates for the lth effect
    resid_l = resid + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    # update prior
    prior, _ = _update_lth_params(resid_l, X, post, prior, l_index)

    # update posterior parameters for lth effect and jth SNP
    _, post = _update_lth_params(resid_l, X, post, prior, l_index)

    # update prior
    # prior = _update_prior_covar(Y, X, post, prior, l_index)

    # update residual for next \ell effect
    resid = resid_l - X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    return _LResult(resid, X, Y, post, prior)


def _fit_model(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams
) -> tuple[Array, PosteriorParams, PriorParams]:
    L, p, k = post.mean_b.shape

    # Compute residuals and update model
    resid = Y - X @ jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)

    init_params = _LResult(resid, X, Y, post, prior)

    _, _, _, post, prior = lax.fori_loop(0, L, _fit_lth_effect, init_params)

    # update prior (i.e. resid_var)
    prior = _update_resid_covar(Y, X, post, prior)

    # compute ELBO
    elbo = _compute_elbo(Y, X, post, prior)

    return elbo, post, prior


def _reorder_l_r(priors: PriorParams, posteriors: PosteriorParams) -> Tuple[PriorParams, PosteriorParams, Array]:
    frob_norm = jnp.sum(jnp.linalg.svd(posteriors.var_b, compute_uv=False), axis=1)

    # we want to reorder them based on the Frobenius norm
    l_order = jnp.argsort(-frob_norm)
    l_order = l_order.reshape(-1)

    new_priors = PriorParams(
        resid_var=priors.resid_var,  # Assuming resid_var does not need reordering
        prob=priors.prob,  # Assuming prob does not need reordering
        var_b=priors.var_b[l_order, :],  # Only var_b is reordered
    )

    new_posteriors = PosteriorParams(
        prob=posteriors.prob[l_order, :], mean_b=posteriors.mean_b[l_order, :], var_b=posteriors.var_b[l_order, :]
    )

    return new_priors, new_posteriors, l_order


def _reorder_l(prior: PriorParams, post: PosteriorParams) -> Tuple[PriorParams, PosteriorParams, Array]:
    frob_norm = jnp.sum(jnp.linalg.svd(prior.var_b, compute_uv=False), axis=1)

    # we want to reorder them based on the Frobenius norm
    l_order = jnp.argsort(-frob_norm)

    # priors effect_covar
    prior = prior._replace(var_b=prior.var_b[l_order])

    post = post._replace(prob=post.prob[l_order], mean_b=post.mean_b[l_order], var_b=post.var_b[l_order])

    return prior, post, l_order


def make_pip(alpha: ArrayLike) -> Array:
    """The function to calculate posterior inclusion probability (PIP).

    Args:
        alpha: :math:`L \\times p` matrix that contains posterior probability for SNP to be causal
            (i.e., :math:`\\alpha` in :ref:`Model`).

    Returns:
        :py:obj:`Array`: :math:`p \\times 1` vector for the posterior inclusion probability.

    """

    pip = -jnp.expm1(jnp.sum(jnp.log1p(-alpha), axis=0))

    return pip


def make_cs(
    alpha: ArrayLike,
    X: ArrayLike,
    threshold: float = 0.9,
    purity: float = 0.5,
    max_select: int = 500,
    seed: int = 12345,
) -> Tuple[pd.DataFrame, pd.DataFrame, Array, Array]:
    """
    Args:

        alpha: L by p matrix contains posterior probability for SNP to be causal
        X: genotype matrix
        N: sample size
        purity: the minimum pairwise correlation across SNPs to be eligible as output credible set
        max_select: the maximum number of selected SNP to compute purity

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame]`: A tuple of
            #. credible set (:py:obj:`pd.DataFrame`) after pruning for purity,
            #. full credible set (:py:obj:`pd.DataFrame`) before pruning for purity.
            #. PIPs (:py:obj:`Array`) across :math:`L` credible sets.
            #. PIPs (:py:obj:`Array`) across credible sets that are not pruned. An array of zero if all credible sets
                are pruned.

    """
    rng_key = rdm.PRNGKey(seed)
    L, p = alpha.shape
    N, _ = X.shape
    t_alpha = pd.DataFrame(alpha.T).reset_index()

    cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "alpha", "c_alpha"])
    full_alphas = t_alpha[["index"]]

    for ldx in range(L):
        tmp_pd = t_alpha[["index", ldx]].sort_values(ldx, ascending=False).reset_index(drop=True)
        tmp_pd["csum"] = tmp_pd[[ldx]].cumsum()
        n_row = tmp_pd[tmp_pd.csum < threshold].shape[0]

        if n_row == tmp_pd.shape[0]:
            select_idx = jnp.arange(n_row)
        else:
            select_idx = jnp.arange(n_row + 1)

        # output CS index is 1-based
        tmp_cs = (
            tmp_pd.iloc[select_idx, :]
            .assign(CSIndex=(ldx + 1))
            .rename(columns={"csum": "c_alpha", "index": "SNPIndex", ldx: "alpha"})
        )

        tmp_pd["in_cs"] = (tmp_pd.index.values <= jnp.max(select_idx)) * 1

        # prepare alphas table's entries
        tmp_pd = tmp_pd.drop(["csum"], axis=1).rename(
            columns={
                "in_cs": f"in_cs_l{ldx + 1}",
                ldx: f"alpha_l{ldx + 1}",
            }
        )

        full_alphas = full_alphas.merge(tmp_pd, how="left", on="index")

        # check the purity
        snp_idx = tmp_cs.SNPIndex.values.astype("int64")

        # randomly select 'max_select' SNPs
        if len(snp_idx) > max_select:
            snp_idx = rdm.choice(rng_key, snp_idx, shape=(max_select,), replace=False)

        # update genotype data and LD
        ld_X = X[:, snp_idx]
        ld = jnp.einsum("jk, jm->km", ld_X, ld_X) / N

        avg_corr = jnp.sum(jnp.min(jnp.abs(ld), axis=(0, 1)))

        full_alphas[f"purity_l{ldx + 1}"] = avg_corr

        if avg_corr > purity:
            cs = pd.concat([cs, tmp_cs], ignore_index=True)
            full_alphas[f"kept_l{ldx + 1}"] = 1
        else:
            full_alphas[f"kept_l{ldx + 1}"] = 0

    pip_all = make_pip(alpha)

    # CSIndex is now 1-based
    pip_cs = make_pip(alpha[(cs.CSIndex.unique().astype(int) - 1),])

    n_snp_cs = cs.SNPIndex.values.astype(int)
    n_snp_cs_unique = jnp.unique(cs.SNPIndex.values.astype(int))

    if len(n_snp_cs) != len(n_snp_cs_unique):
        log.logger.warning(
            "Same SNPs appear in different credible set, which is very unusual."
            + " You may want to check this gene in details."
        )

    cs["pip_all"] = jnp.array([pip_all[idx] for idx in cs.SNPIndex.values.astype(int)])
    cs["pip_cs"] = jnp.array([pip_cs[idx] for idx in cs.SNPIndex.values.astype(int)])

    full_alphas["pip_all"] = pip_all
    full_alphas["pip_cs"] = pip_cs
    full_alphas = full_alphas.rename(columns={"index": "SNPIndex"})

    # log.logger.info(
    # f"{len(cs.CSIndex.unique())} out of {L} credible sets remain after pruning based on purity ({purity})."
    # + " For detailed results, specify --alphas."
    # )

    return cs, full_alphas, pip_all, pip_cs


# Define calculate lsfr


def cal_pos_prob(post: PosteriorParams) -> Array:
    # part a
    # alpha is in the shape (L, p)
    # extend part_a to be the shape (L, p, 1) for broadcasting
    part_a = jnp.log(post.prob[:, :, jnp.newaxis])

    # part b
    posterior_std = jnp.sqrt(jnp.diagonal(post.var_b, axis1=-2, axis2=-1))
    # calculate z-scores
    z_scores = post.mean_b / posterior_std
    # calculate positive probabilities
    part_b = stats.norm.logsf(z_scores)
    # computing the log of the survival function (complementary CDF)
    # z_scores is in the shape (L, p, k)
    # z score for in each effect each SNP on each cell type
    tmp = part_a + part_b
    # calculate the min
    pos_prob = jnp.exp(tmp)
    # post_prob will be in the shape (L, p, k)
    return pos_prob


def cal_neg_prob(post: PosteriorParams) -> Array:
    # part a
    # alpha is in the shape (L, p)
    # extend part_a to be the shape (L, p, 1) for broadcasting
    part_a = jnp.log(post.prob[:, :, jnp.newaxis])

    # part b
    posterior_std = jnp.sqrt(jnp.diagonal(post.var_b, axis1=-2, axis2=-1))
    # calculate z-scores
    z_scores = post.mean_b / posterior_std
    # calculate negative probabilities
    part_b = stats.norm.logcdf(z_scores)
    # z_scores is in the shape (L, p, k)
    # z score for in each effect each SNP on each cell type
    tmp = part_a + part_b
    neg_prob = jnp.exp(tmp)
    # neg_prob will be in the shape (L, p, k)
    return neg_prob


def cal_lfsr(post: PosteriorParams) -> Array:
    # calculate lfdr (local false discovery rate)
    lfdr = 1 - post.prob
    # call pos and neg probs
    pos_prob = cal_pos_prob(post)
    neg_prob = cal_neg_prob(post)
    # extend the dimension for broadcasting
    zero_prob = lfdr[:, :, jnp.newaxis]

    non_neg_prob = zero_prob + pos_prob

    non_pos_prob = zero_prob + neg_prob

    min_non_neg_prob = jnp.min(non_neg_prob, axis=0)
    min_non_pos_prob = jnp.min(non_pos_prob, axis=0)

    lfsr = jnp.minimum(min_non_neg_prob, min_non_pos_prob)
    # lsfr will be in the shape (p, k)
    # representing the lfsr for each SNP on each cell type
    return lfsr


def finemap(
    Y: ArrayLike,
    X: ArrayLike,
    L: int,
    prior_var: float = 1e-3,
    tol: float = 1e-3,
    max_iter: int = 100,
    no_reorder: bool = False,
):
    # todo: QC the input data; check dimensions match, etc.
    # check dimensions match
    n_y, k = Y.shape
    n_x, p = X.shape
    if n_y != n_x:
        raise ValueError("Number of individuals do not match: " f"Y is {n_y}x{k}, but X is {n_x}x{p}")

    # initialize model parameters
    prior = PriorParams(
        resid_var=jnp.eye(k),
        prob=jnp.ones(p) / p,
        var_b=jnp.tile(prior_var * jnp.eye(k), (L, 1, 1)),
    )

    post = PosteriorParams(
        mean_b=jnp.zeros((L, p, k)),
        var_b=jnp.eye(k) + jnp.zeros((L, p, k, k)),
        prob=jnp.tile(prior.prob, (L, 1)),
    )

    # evaluate current elbo
    elbo = -jnp.inf
    elbo_increase = True
    for train_iter in range(max_iter):
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        print(f"iteration: {train_iter}, prior: {prior}")
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        print("We are using SCFM Jun 20 Version!")
        if jnp.fabs(cur_elbo - elbo) < tol:
            print(f"ELBO has converged. ELBO at the last iteration: {cur_elbo}")
            break

        # Update the last ELBO to the current ELBO at the end of the iteration
        elbo = cur_elbo

    l_order = jnp.arange(L)
    if not no_reorder:
        prior, post, l_order = _reorder_l(prior, post)

    # Fine-mapping
    cs, full_alphas, pip_all, pip_cs = make_cs(
        post.prob,
        X,
        threshold=0.9,
        purity=0.5,
        max_select=500,
        seed=12345,
    )

    # Calculate lsfr
    lfsr = cal_lfsr(post)

    return SCFMResult(prior, post, pip_all, pip_cs, cs, full_alphas, elbo, elbo_increase, l_order, lfsr)
