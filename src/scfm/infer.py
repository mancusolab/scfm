from functools import partial
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
    snp_lfsr: the local false sign rate at SNP for each cell type in CS
    cs_lfsr: the local false sign rate for each CS 
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
    snp_lfsr: pd.DataFrame
    cs_lfsr: pd.DataFrame


@jax.jit
def _update_lth_params(
    R_l: Array, X: Array, post: PosteriorParams, prior: PriorParams, l_index: int
) -> tuple[PriorParams, PosteriorParams]:
    """
    Update the parameters for the l-th effect
    
    Args:
        R_l: Residual matrix
        X: Design matrix
        post: Posterior parameters
        prior: Prior parameters
        l_index: Index of the effect to update
        
    Returns:
        Updated prior and posterior parameters
    """
    n, k = R_l.shape
    n, p = X.shape

    # Precompute sum of squares for each SNP
    XtX = jnp.sum(X * X, axis=0)  # Shape: (p,)

    # Calculate precisions from covariances
    prior_prec = jnpla.inv(prior.var_b[l_index])  # Shape: (k,k)
    resid_prec = jnpla.inv(prior.resid_var)       # Shape: (k,k)

    # Calculate posterior covariance for each SNP
    post_cov = jnpla.inv(resid_prec * XtX[:, jnp.newaxis, jnp.newaxis] + prior_prec)  # Shape: (p,k,k)
    
    # Calculate X^T R Σ^-1 term
    rTXSigInv = (resid_prec @ R_l.T) @ X  # Shape: (k,p)
    
    # Calculate posterior mean for each SNP
    post_mean = jnp.einsum("pkq,qp->pk", post_cov, rTXSigInv)  # Shape: (p,k)

    # Calculate posterior inclusion probabilities
    alpha = jax.nn.softmax(
        jnp.log(prior.prob) - stats.multivariate_normal.logpdf(post_mean, jnp.zeros(k), post_cov)
    )  # Shape: (p,)

    # Update prior effect size covariance
    post_mean_sq = jnp.einsum("pk,pq->pkq", post_mean, post_mean) + post_cov  # Shape: (p,k,k)
    prior_covar_b = jnp.einsum("p,pkq->kq", alpha, post_mean_sq)  # Shape: (k,k)

    # Update prior parameters
    prior = prior._replace(
        var_b=prior.var_b.at[l_index].set(prior_covar_b),
    )

    # Update posterior parameters
    post = PosteriorParams(
        prob=post.prob.at[l_index].set(alpha),
        mean_b=post.mean_b.at[l_index, :, :].set(post_mean),
        var_b=post.var_b.at[l_index].set(post_cov),
    )

    return prior, post


@jax.jit
def _update_resid_covar(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> PriorParams:
    """
    Update the residual covariance matrix based on current posteriors
    
    Args:
        Y: Observed phenotype matrix
        X: Design matrix
        post: Posterior parameters
        prior: Prior parameters
        
    Returns:
        Updated prior parameters with new residual variance
    """
    n, p = X.shape

    # Calculate expected effect sizes across all L effects
    # E_Q(B) = sum_l alpha_l * mu_l
    B = jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)  # Shape: (p,k)

    # Calculate predicted values
    pred = X @ B  # Shape: (n,k)
    
    # Calculate second moment of effects
    # E_Q[B^T B] includes both mean and variance components
    M = jnp.einsum("lpk,lpq,lp->pkq", post.mean_b, post.mean_b, post.prob) + jnp.einsum(
        "lp,lpkq->pkq", post.prob, post.var_b
    )  # Shape: (p,k,k)
    
    # Calculate X^T X weighted by second moment
    outer_moment = jnp.einsum("nj,nj,jkq->kq", X, X, M) + pred.T @ pred  # Shape: (k,k)

    # Compute terms for residual variance update
    term1 = Y.T @ Y                 # Y^T Y
    term2 = -2 * Y.T @ pred         # -2Y^T X E[B]
    term3 = outer_moment            # E[B^T X^T X B]

    # Updated residual variance
    resid_var = (term1 + term2 + term3) / n

    # Create updated prior
    prior = prior._replace(
        resid_var=resid_var,
    )

    return prior


@jax.jit
def expected_loglikelihood(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> float:
    """
    Calculate the expected log-likelihood given current parameters
    
    Args:
        Y: Observed phenotype matrix
        X: Design matrix
        post: Posterior parameters
        prior: Prior parameters
        
    Returns:
        Expected log-likelihood value
    """
    n, p = X.shape
    L, p, k = post.mean_b.shape

    # Calculate expected effect sizes: E_Q(B)
    B = jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)  # Shape: (p,k)

    # Calculate predictions
    pred = X @ B  # Shape: (n,k)
    
    # Calculate second moments including both means and variances
    M = jnp.einsum("lpk,lpq->lpkq", post.mean_b, post.mean_b) + post.var_b  # Shape: (L,p,k,k)
    
    # Calculate outer moment term
    outer_moment = jnp.einsum("nj,nj,lj,ljkq->kq", X, X, post.prob, M) + pred.T @ pred  # Shape: (k,k)
    
    # Solve linear systems with residual precision matrix
    inv_prec_Yt = jspla.solve(prior.resid_var, Y.T, assume_a="pos")  # Shape: (k,n)
    inv_prec_outer_moment = jspla.solve(prior.resid_var, outer_moment, assume_a="pos")  # Shape: (k,k)
    
    # Calculate log-likelihood
    ll = -0.5 * (
        (
            jnp.sum(inv_prec_Yt * Y.T)               # tr(Σ⁻¹ Y'Y)
            - 2 * jnp.sum(inv_prec_Yt * pred.T)      # -2tr(Σ⁻¹ Y'X E[B])
            + jnp.trace(inv_prec_outer_moment)       # tr(Σ⁻¹ E[B'X'XB])
        )
        + k * jnp.log(2 * jnp.pi)                   # Constant term from multivariate normal
        + n * jnpla.slogdet(prior.resid_var)[1]     # Log determinant term
    )

    return ll


@jax.jit
def _compute_elbo(Y: Array, X: Array, post: PosteriorParams, prior: PriorParams) -> Array:
    """
    Compute Evidence Lower Bound (ELBO) for variational inference
    
    Args:
        Y: Observed phenotype matrix
        X: Design matrix
        post: Posterior parameters
        prior: Prior parameters
        
    Returns:
        Current ELBO value
    """
    # Compute expected log-likelihood
    ll = expected_loglikelihood(Y, X, post, prior)
    
    # Compute KL divergence between posterior and prior
    kl = kl_single_effect(prior, post)

    # ELBO = Expected log-likelihood - KL divergence
    cur_elbo = ll - kl

    return cur_elbo


@partial(jax.jit, static_argnums=(0,))
def _fit_lth_effect(l_index: int, params: _LResult) -> _LResult:
    resid, X, Y, post, prior = params

    # Add back the contribution of the l-th effect to get the current residual
    resid_l = resid + X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    # Update prior parameters for the l-th effect
    prior, _ = _update_lth_params(resid_l, X, post, prior, l_index)

    # Update posterior parameters for the l-th effect using the updated prior
    _, post = _update_lth_params(resid_l, X, post, prior, l_index)

    # Update residual for next effect by subtracting the contribution of the current effect
    resid = resid_l - X @ (post.mean_b[l_index, :, :] * post.prob[l_index, :, jnp.newaxis])

    return _LResult(resid, X, Y, post, prior)


@jax.jit
def _fit_model(
    Y: Array, X: Array, post: PosteriorParams, prior: PriorParams
) -> tuple[Array, PosteriorParams, PriorParams]:
    L, p, k = post.mean_b.shape

    # Compute residuals and update model
    resid = Y - X @ jnp.einsum("lp,lpk->pk", post.prob, post.mean_b)

    init_params = _LResult(resid, X, Y, post, prior)

    # Apply each effect update
    _, _, _, post, prior = lax.fori_loop(0, L, _fit_lth_effect, init_params)

    # update prior (i.e. resid_var)
    prior = _update_resid_covar(Y, X, post, prior)

    # compute ELBO
    elbo = _compute_elbo(Y, X, post, prior)

    return elbo, post, prior


@jax.jit
def _reorder_l(prior: PriorParams, post: PosteriorParams) -> Tuple[PriorParams, PosteriorParams, Array]:
    frob_norm = jnp.sum(jnp.linalg.svd(prior.var_b, compute_uv=False), axis=1)

    # we want to reorder them based on the Frobenius norm
    l_order = jnp.argsort(-frob_norm)

    # priors effect_covar
    prior = prior._replace(var_b=prior.var_b[l_order])

    post = post._replace(prob=post.prob[l_order], mean_b=post.mean_b[l_order], var_b=post.var_b[l_order])

    return prior, post, l_order


@jax.jit
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
    Create credible sets from posterior probabilities
    
    Args:
        alpha: L by p matrix contains posterior probability for SNP to be causal
        X: genotype matrix
        threshold: probability threshold for credible set inclusion (default 0.9)
        purity: the minimum pairwise correlation across SNPs to be eligible as output credible set
        max_select: the maximum number of selected SNP to compute purity
        seed: random seed for reproducibility

    Returns:
        :py:obj:`Tuple[pd.DataFrame, pd.DataFrame, Array, Array]`: A tuple of
            1. credible set (:py:obj:`pd.DataFrame`) after pruning for purity
            2. full credible set (:py:obj:`pd.DataFrame`) before pruning for purity
            3. PIPs (:py:obj:`Array`) across all L credible sets
            4. PIPs (:py:obj:`Array`) across credible sets that are not pruned
    """
    # Initialize random key
    rng_key = rdm.PRNGKey(seed)
    L, p = alpha.shape
    N, _ = X.shape
    
    # Convert alpha to a dataframe for easier processing
    alpha_np = jnp.asarray(alpha).T  # Make sure it's a JAX array and transpose
    t_alpha = pd.DataFrame(alpha_np)
    t_alpha['index'] = jnp.arange(p)
    
    # Initialize output dataframes
    cs_list = []
    full_alphas = pd.DataFrame({'SNPIndex': jnp.arange(p)})
    
    # Pre-calculate PIPs to avoid redundant computation
    pip_all = make_pip(alpha)
    
    # Process each credible set
    for ldx in range(L):
        # Sort SNPs by their alpha values for this effect
        tmp_df = pd.DataFrame({
            'SNPIndex': jnp.arange(p),
            'alpha': alpha_np[:, ldx],
        }).sort_values('alpha', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative sum and find threshold crossing
        tmp_df['c_alpha'] = tmp_df['alpha'].cumsum()
        n_row = tmp_df[tmp_df['c_alpha'] < threshold].shape[0]
        
        # Handle edge case where all SNPs are needed
        last_idx = n_row if n_row == tmp_df.shape[0] else n_row + 1
        
        # Create CS for this effect (1-based index)
        tmp_cs = tmp_df.iloc[:last_idx].copy()
        tmp_cs['CSIndex'] = ldx + 1
        
        # Update tracking for full alphas dataframe
        in_cs = pd.Series(0, index=range(p))
        in_cs.iloc[tmp_df.iloc[:last_idx]['SNPIndex'].values] = 1
        full_alphas[f"in_cs_l{ldx + 1}"] = in_cs.values
        full_alphas[f"alpha_l{ldx + 1}"] = alpha_np[:, ldx]
        
        # For purity calculation, convert to integer indices
        snp_idx = tmp_cs['SNPIndex'].values.astype("int64")
        
        # Check if we need to subsample SNPs for purity calculation
        if len(snp_idx) > max_select:
            rng_key, subkey = rdm.split(rng_key)
            snp_idx = rdm.choice(subkey, snp_idx, shape=(max_select,), replace=False)
        
        # Calculate linkage disequilibrium and purity
        if len(snp_idx) > 1:  # Only calculate purity if we have multiple SNPs
            ld_X = X[:, snp_idx]
            # Optimize LD calculation for efficiency
            ld = jnp.einsum("jk,jm->km", ld_X, ld_X) / N
            # Minimum absolute correlation as purity measure
            avg_corr = jnp.min(jnp.abs(ld - jnp.eye(ld.shape[0])) + jnp.eye(ld.shape[0]))
        else:
            # If only one SNP, purity is perfect
            avg_corr = 1.0
            
        full_alphas[f"purity_l{ldx + 1}"] = avg_corr
        
        # Add to CS if purity threshold is met
        if avg_corr > purity:
            cs_list.append(tmp_cs)
            full_alphas[f"kept_l{ldx + 1}"] = 1
        else:
            full_alphas[f"kept_l{ldx + 1}"] = 0
    
    # Combine all credible sets
    if cs_list:
        cs = pd.concat(cs_list, ignore_index=True)
        # Calculate PIP for credible sets that passed purity
        cs_indices = jnp.array(cs['CSIndex'].unique().astype(int) - 1)
        pip_cs = make_pip(alpha[cs_indices]) if len(cs_indices) > 0 else jnp.zeros_like(pip_all)
        
        # Add PIPs to credible sets
        cs['pip_all'] = cs['SNPIndex'].map(lambda idx: pip_all[idx])
        cs['pip_cs'] = cs['SNPIndex'].map(lambda idx: pip_cs[idx])
        
        # Check for duplicated SNPs across credible sets
        n_snp_cs = cs.SNPIndex.values.astype(int)
        n_snp_cs_unique = jnp.unique(n_snp_cs)
        if len(n_snp_cs) != len(n_snp_cs_unique):
            log.logger.warning(
                "Same SNPs appear in different credible set, which is very unusual."
                + " You may want to check this gene in details."
            )
    else:
        # Create empty dataframe with correct columns if no CS passed purity
        cs = pd.DataFrame(columns=["CSIndex", "SNPIndex", "alpha", "c_alpha", "pip_all", "pip_cs"])
        pip_cs = jnp.zeros_like(pip_all)
    
    # Add PIPs to full alphas dataframe
    full_alphas["pip_all"] = pip_all
    full_alphas["pip_cs"] = pip_cs
    
    return cs, full_alphas, pip_all, pip_cs


# Define calculate lsfr
@jax.jit
def _compute_clfsr(post: PosteriorParams) -> Array:
    # pull marginal posterior SDs
    posterior_std = jnp.sqrt(jnp.diagonal(post.var_b, axis1=-2, axis2=-1))

    # calculate negative probabilities
    cdf = stats.norm.cdf(0.0, loc=post.mean_b, scale=posterior_std)
    sf = stats.norm.sf(0.0, loc=post.mean_b, scale=posterior_std)

    # 1 - max(p, 1 - p) == min(p, 1 - p)
    return jnp.minimum(sf, cdf)


# Function to check if precompilation has been done
def _check_precompilation():
    import os
    flag_file = os.path.expanduser("~/.scfm/precompiled")
    return os.path.exists(flag_file)


def finemap(
    Y: ArrayLike,
    X: ArrayLike,
    L: int,
    prior_var: float = 1e-3,
    tol: float = 1e-3,
    max_iter: int = 100,
    no_reorder: bool = False,
    precompile: bool = True,
    prior_covar_filter: float = None,
):
    """
    Perform fine-mapping inference using SCFM
    
    Args:
        Y: Phenotype matrix of shape (n_samples, n_traits)
        X: Genotype matrix of shape (n_samples, n_snps)
        L: Number of causal effects to model
        prior_var: Initial prior variance for effect sizes
        tol: Convergence tolerance for ELBO
        max_iter: Maximum number of iterations
        no_reorder: If True, skip reordering of effects by magnitude
        precompile: If True (default), perform precompilation when needed to improve performance
                    Set to False to skip precompilation (useful for batch jobs)
        prior_covar_filter: If provided, filter effects where log(trace_norm_l1) - log(trace_norm_lj) > threshold
        
    Returns:
        SCFMResult containing posterior estimates and credible sets
    """
    # Handle precompilation (turned on by default)
    if precompile:
        if not _check_precompilation():
            try:
                print("First-time run: performing precompilation to speed up this and future runs...")
                from .precompile import precompile_scfm
                precompile_scfm()
            except Exception as e:
                print(f"Warning: Precompilation failed: {e}")
                print("Continuing without precompilation. Future performance may be slower.")
    
    # Check dimensions match
    n_y, k = Y.shape
    n_x, p = X.shape
    if n_y != n_x:
        raise ValueError("Number of individuals do not match: " f"Y is {n_y}x{k}, but X is {n_x}x{p}")

    # Initialize model parameters
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

    # Set up for ELBO tracking and convergence
    elbo = -jnp.inf
    elbo_increase = True
    
    # Profiling timers
    import time
    total_iter_time = 0
    
    # Main inference loop
    for train_iter in range(max_iter):
        iter_start = time.time()
        
        # Update model parameters
        cur_elbo, post, prior = _fit_model(Y, X, post, prior)
        
        # Track timing
        iter_end = time.time()
        iter_time = iter_end - iter_start
        total_iter_time += iter_time
        
        # Report progress
        print(f"iteration: {train_iter}, time: {iter_time:.4f}s, prior: {prior}")
        print(f"ELBO[{train_iter}] = {cur_elbo}")
        
        # Check for convergence
        if jnp.fabs(cur_elbo - elbo) < tol:
            print(f"ELBO has converged. ELBO at the last iteration: {cur_elbo}")
            break

        # Update for next iteration
        elbo = cur_elbo

    # Report timing summary
    print(f"Total iteration time: {total_iter_time:.4f}s, Average: {total_iter_time/(train_iter+1):.4f}s per iteration")
    
    # Reorder effects by magnitude
    reorder_start = time.time()
    l_order = jnp.arange(L)
    if not no_reorder:
        prior, post, l_order = _reorder_l(prior, post)
    reorder_time = time.time() - reorder_start
    print(f"Reordering time: {reorder_time:.4f}s")

    # Apply prior covariance filtering if threshold is provided
    if prior_covar_filter is not None:
        # Compute trace norms for all effects
        trace_norms = jnp.array([jnp.trace(prior.var_b[l]) for l in range(L)])
        log_trace_norms = jnp.log(trace_norms)
        
        # Calculate differences from the first effect
        log_trace_diff = log_trace_norms[0] - log_trace_norms
        
        # Find effects that pass the threshold
        valid_effects = jnp.where(log_trace_diff <= prior_covar_filter)[0]
        
        print(f"Prior covariance filtering: keeping {len(valid_effects)} out of {L} effects")
        print(f"Log trace norm differences: {log_trace_diff}")
        
        # Filter prior and posterior parameters
        prior = prior._replace(var_b=prior.var_b[valid_effects])
        post = post._replace(
            prob=post.prob[valid_effects],
            mean_b=post.mean_b[valid_effects],
            var_b=post.var_b[valid_effects]
        )
        
        # Update l_order to reflect filtering
        l_order = l_order[valid_effects]
    
    # Create credible sets
    cs_start = time.time()
    cs, full_alphas, pip_all, pip_cs = make_cs(
        post.prob,
        X,
        threshold=0.9,
        purity=0.5,
        max_select=500,
        seed=12345,
    )
    cs_time = time.time() - cs_start
    print(f"Credible set computation time: {cs_time:.4f}s")

    # Calculate LFSR (local false sign rate)
    lfsr_start = time.time()
    
    # Compute LFSR for all SNPs and traits
    clfsr = _compute_clfsr(post)
    min_lfsr = jnp.min(1.0 - post.prob[:, :, jnp.newaxis] * (1.0 - clfsr), axis=0)

    # Create SNP-level LFSR results
    if not cs.empty:
        # Process LFSR for SNPs in credible sets
        snp_lfsr = pd.DataFrame({"SNPIndex": cs["SNPIndex"].unique()})
        snp_indices = snp_lfsr["SNPIndex"].to_numpy()
        matched_rows = jnp.array([min_lfsr[i] for i in snp_indices])
        column_names = [f"celltype{i}" for i in range(1, k + 1)]
        matched_rows_df = pd.DataFrame(matched_rows, columns=column_names)
        snp_lfsr = pd.concat([snp_lfsr, matched_rows_df], axis=1)
        
        # Compute credible set level LFSR
        cs_lfsr_tmp = pd.merge(cs, snp_lfsr, on="SNPIndex")
        weighted_cols = []
        for celltype in column_names:
            cs_lfsr_tmp[f"weighted_{celltype}"] = cs_lfsr_tmp["alpha"] * cs_lfsr_tmp[celltype]
            weighted_cols.append(f"weighted_{celltype}")
            
        # Group by credible set
        cs_lfsr = cs_lfsr_tmp.groupby("CSIndex")[weighted_cols].sum()
        cs_lfsr = cs_lfsr.rename(columns=lambda col: col.replace("weighted_", ""))
        cs_lfsr.reset_index(inplace=True)
    else:
        # Handle case with no credible sets
        snp_lfsr = pd.DataFrame(columns=["SNPIndex"] + [f"celltype{i}" for i in range(1, k + 1)])
        cs_lfsr = pd.DataFrame(columns=["CSIndex"] + [f"celltype{i}" for i in range(1, k + 1)])
    
    lfsr_time = time.time() - lfsr_start
    print(f"LFSR computation time: {lfsr_time:.4f}s")
    
    # Return final results    
    return SCFMResult(prior, post, pip_all, pip_cs, cs, full_alphas, elbo, elbo_increase, l_order, min_lfsr, snp_lfsr, cs_lfsr)
