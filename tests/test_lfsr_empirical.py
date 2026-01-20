"""
Empirical validation of LFSR (Local False Sign Rate) computation.

This test validates that the analytical LFSR computation matches empirical
estimates from sampling from the posterior distribution.

LFSR definitions from the paper:
- clfsr (conditional LFSR): 1 - max{Pr(b > 0 | X, Y, γ = 1), Pr(b < 0 | X, Y, γ = 1)}
- lfsr (unconditional LFSR): 1 - max{Pr(b > 0 | X, Y), Pr(b < 0 | X, Y)}
                            = 1 - α * (1 - clfsr)
- min-lfsr: minimum lfsr across all single effects L
"""

import jax
import jax.numpy as jnp
import jax.random as rdm
import jax.scipy.stats as stats
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)


def compute_analytical_clfsr(mean_b: Array, var_b: Array) -> Array:
    """
    Compute analytical conditional LFSR.

    clfsr = 1 - max{Pr(b > 0 | γ = 1), Pr(b < 0 | γ = 1)}
          = min{Pr(b ≤ 0 | γ = 1), Pr(b ≥ 0 | γ = 1)}
          = min{CDF(0), SF(0)}

    Args:
        mean_b: Posterior mean of shape (L, p, k)
        var_b: Posterior covariance of shape (L, p, k, k)

    Returns:
        clfsr of shape (L, p, k)
    """
    # Get marginal posterior standard deviations
    posterior_std = jnp.sqrt(jnp.diagonal(var_b, axis1=-2, axis2=-1))  # (L, p, k)

    # Calculate Pr(b ≤ 0) and Pr(b > 0)
    cdf = stats.norm.cdf(0.0, loc=mean_b, scale=posterior_std)  # Pr(b ≤ 0)
    sf = stats.norm.sf(0.0, loc=mean_b, scale=posterior_std)     # Pr(b > 0)

    # clfsr = min(cdf, sf)
    return jnp.minimum(cdf, sf)


def compute_analytical_lfsr(alpha: Array, clfsr: Array) -> Array:
    """
    Compute analytical unconditional LFSR.

    lfsr = 1 - max{Pr(b > 0), Pr(b < 0)}
         = 1 - α * (1 - clfsr)

    Args:
        alpha: Posterior inclusion probability of shape (L, p)
        clfsr: Conditional LFSR of shape (L, p, k)

    Returns:
        lfsr of shape (L, p, k)
    """
    # alpha has shape (L, p), clfsr has shape (L, p, k)
    # Need to broadcast alpha to (L, p, k)
    return 1.0 - alpha[:, :, jnp.newaxis] * (1.0 - clfsr)


def compute_analytical_min_lfsr(lfsr: Array) -> Array:
    """
    Compute min-lfsr across single effects.

    min_lfsr = min_l(lfsr_l)

    Args:
        lfsr: LFSR of shape (L, p, k)

    Returns:
        min_lfsr of shape (p, k)
    """
    return jnp.min(lfsr, axis=0)


def sample_empirical_clfsr(
    mean_b: Array,
    var_b: Array,
    n_samples: int,
    key: rdm.PRNGKey
) -> Array:
    """
    Compute empirical conditional LFSR by sampling.

    Sample from N(μ, Σ) and compute the fraction of samples on each side of zero.
    clfsr = min{empirical Pr(b ≤ 0), empirical Pr(b > 0)}

    Args:
        mean_b: Posterior mean of shape (L, p, k)
        var_b: Posterior covariance of shape (L, p, k, k)
        n_samples: Number of Monte Carlo samples
        key: JAX random key

    Returns:
        Empirical clfsr of shape (L, p, k)
    """
    L, p, k = mean_b.shape

    # Sample independently for each (l, j) pair
    # For multivariate normal, we need to sample from N(μ, Σ) for each (l, j)
    # var_b has shape (L, p, k, k)

    keys = rdm.split(key, L * p)
    keys = keys.reshape(L, p, 2)

    empirical_clfsr = jnp.zeros((L, p, k))

    for l in range(L):
        for j in range(p):
            subkey = keys[l, j]
            # Sample from multivariate normal
            samples = rdm.multivariate_normal(
                subkey,
                mean=mean_b[l, j],
                cov=var_b[l, j],
                shape=(n_samples,)
            )  # (n_samples, k)

            # Compute fraction of samples on each side of zero
            pr_positive = jnp.mean(samples > 0, axis=0)  # (k,)
            pr_negative = jnp.mean(samples <= 0, axis=0)  # (k,)

            # clfsr = min(pr_positive, pr_negative)
            # But actually: clfsr = 1 - max(pr_positive, 1 - pr_positive)
            # which equals min(pr_positive, 1 - pr_positive) = min(pr_positive, pr_negative)
            empirical_clfsr = empirical_clfsr.at[l, j].set(jnp.minimum(pr_positive, pr_negative))

    return empirical_clfsr


def sample_empirical_lfsr(
    alpha: Array,
    mean_b: Array,
    var_b: Array,
    n_samples: int,
    key: rdm.PRNGKey
) -> Array:
    """
    Compute empirical unconditional LFSR by sampling.

    With probability α, sample from N(μ, Σ); otherwise b = 0.
    Then compute: lfsr = 1 - max{Pr(b > 0), Pr(b < 0)}

    Args:
        alpha: Posterior inclusion probability of shape (L, p)
        mean_b: Posterior mean of shape (L, p, k)
        var_b: Posterior covariance of shape (L, p, k, k)
        n_samples: Number of Monte Carlo samples
        key: JAX random key

    Returns:
        Empirical lfsr of shape (L, p, k)
    """
    L, p, k = mean_b.shape

    keys = rdm.split(key, L * p * 2)
    keys = keys.reshape(L, p, 2, 2)

    empirical_lfsr = jnp.zeros((L, p, k))

    for l in range(L):
        for j in range(p):
            include_key = keys[l, j, 0]
            sample_key = keys[l, j, 1]

            # Decide which samples include this effect
            include_mask = rdm.uniform(include_key, shape=(n_samples,)) < alpha[l, j]

            # Sample effect sizes from N(μ, Σ)
            effect_samples = rdm.multivariate_normal(
                sample_key,
                mean=mean_b[l, j],
                cov=var_b[l, j],
                shape=(n_samples,)
            )  # (n_samples, k)

            # Apply inclusion: b = effect if included, else 0
            samples = jnp.where(include_mask[:, jnp.newaxis], effect_samples, 0.0)

            # Compute fraction of samples on each side of zero
            pr_positive = jnp.mean(samples > 0, axis=0)  # (k,)
            pr_negative = jnp.mean(samples < 0, axis=0)  # (k,)

            # lfsr = 1 - max(pr_positive, pr_negative)
            empirical_lfsr = empirical_lfsr.at[l, j].set(1.0 - jnp.maximum(pr_positive, pr_negative))

    return empirical_lfsr


def test_clfsr_matches_empirical():
    """Test that analytical clfsr matches empirical sampling."""
    print("=" * 60)
    print("Testing: Conditional LFSR (clfsr) - analytical vs empirical")
    print("=" * 60)

    key = rdm.PRNGKey(42)

    # Test parameters
    L = 2  # Number of single effects
    p = 10  # Number of SNPs
    k = 3   # Number of traits
    n_samples = 100000

    key, subkey1, subkey2 = rdm.split(key, 3)

    # Generate random posterior parameters
    mean_b = rdm.normal(subkey1, shape=(L, p, k)) * 0.5  # Posterior means

    # Generate positive definite covariance matrices
    var_b = jnp.zeros((L, p, k, k))
    for l in range(L):
        for j in range(p):
            key, subkey = rdm.split(key)
            A = rdm.normal(subkey, shape=(k, k)) * 0.3
            cov = A @ A.T + 0.1 * jnp.eye(k)
            var_b = var_b.at[l, j].set(cov)

    # Compute analytical clfsr
    analytical_clfsr = compute_analytical_clfsr(mean_b, var_b)

    # Compute empirical clfsr
    empirical_clfsr = sample_empirical_clfsr(mean_b, var_b, n_samples, subkey2)

    # Compare
    diff = jnp.abs(analytical_clfsr - empirical_clfsr)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"\nResults with {n_samples:,} samples:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Show a few examples
    print(f"\nSample comparisons (L=0, first 3 SNPs, first 2 traits):")
    for j in range(min(3, p)):
        for r in range(min(2, k)):
            print(f"  SNP {j}, Trait {r}: analytical={analytical_clfsr[0, j, r]:.4f}, "
                  f"empirical={empirical_clfsr[0, j, r]:.4f}, "
                  f"diff={diff[0, j, r]:.4f}")

    # Assert reasonable match (allow for Monte Carlo error)
    tolerance = 0.02  # 2% tolerance
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"
    print(f"\n✓ PASSED: clfsr matches within {tolerance:.1%} tolerance")

    return analytical_clfsr, empirical_clfsr


def test_lfsr_matches_empirical():
    """Test that analytical lfsr matches empirical sampling."""
    print("\n" + "=" * 60)
    print("Testing: Unconditional LFSR (lfsr) - analytical vs empirical")
    print("=" * 60)

    key = rdm.PRNGKey(123)

    # Test parameters
    L = 2  # Number of single effects
    p = 10  # Number of SNPs
    k = 3   # Number of traits
    n_samples = 100000

    key, subkey1, subkey2, subkey3 = rdm.split(key, 4)

    # Generate random posterior parameters
    alpha = rdm.uniform(subkey1, shape=(L, p))  # Inclusion probabilities
    mean_b = rdm.normal(subkey2, shape=(L, p, k)) * 0.5  # Posterior means

    # Generate positive definite covariance matrices
    var_b = jnp.zeros((L, p, k, k))
    for l in range(L):
        for j in range(p):
            key, subkey = rdm.split(key)
            A = rdm.normal(subkey, shape=(k, k)) * 0.3
            cov = A @ A.T + 0.1 * jnp.eye(k)
            var_b = var_b.at[l, j].set(cov)

    # Compute analytical values
    analytical_clfsr = compute_analytical_clfsr(mean_b, var_b)
    analytical_lfsr = compute_analytical_lfsr(alpha, analytical_clfsr)

    # Compute empirical lfsr
    empirical_lfsr = sample_empirical_lfsr(alpha, mean_b, var_b, n_samples, subkey3)

    # Compare
    diff = jnp.abs(analytical_lfsr - empirical_lfsr)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"\nResults with {n_samples:,} samples:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Show a few examples
    print(f"\nSample comparisons (L=0, first 3 SNPs, first 2 traits):")
    for j in range(min(3, p)):
        for r in range(min(2, k)):
            print(f"  SNP {j} (α={alpha[0, j]:.3f}), Trait {r}: "
                  f"analytical={analytical_lfsr[0, j, r]:.4f}, "
                  f"empirical={empirical_lfsr[0, j, r]:.4f}, "
                  f"diff={diff[0, j, r]:.4f}")

    # Assert reasonable match
    tolerance = 0.02  # 2% tolerance
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"
    print(f"\n✓ PASSED: lfsr matches within {tolerance:.1%} tolerance")

    return analytical_lfsr, empirical_lfsr


def test_min_lfsr():
    """Test that min_lfsr correctly takes the minimum across single effects."""
    print("\n" + "=" * 60)
    print("Testing: min-LFSR computation")
    print("=" * 60)

    key = rdm.PRNGKey(456)

    # Test parameters
    L = 3
    p = 5
    k = 2

    key, subkey1, subkey2 = rdm.split(key, 3)

    # Generate random posterior parameters
    alpha = rdm.uniform(subkey1, shape=(L, p))
    mean_b = rdm.normal(subkey2, shape=(L, p, k)) * 0.5

    var_b = jnp.zeros((L, p, k, k))
    for l in range(L):
        for j in range(p):
            key, subkey = rdm.split(key)
            A = rdm.normal(subkey, shape=(k, k)) * 0.3
            cov = A @ A.T + 0.1 * jnp.eye(k)
            var_b = var_b.at[l, j].set(cov)

    # Compute LFSR and min-LFSR
    clfsr = compute_analytical_clfsr(mean_b, var_b)
    lfsr = compute_analytical_lfsr(alpha, clfsr)
    min_lfsr = compute_analytical_min_lfsr(lfsr)

    # Verify min-lfsr is indeed the minimum
    print(f"\nLFSR values for SNP 0, Trait 0 across {L} single effects:")
    for l in range(L):
        print(f"  L={l}: lfsr = {lfsr[l, 0, 0]:.4f}")
    print(f"  min-lfsr = {min_lfsr[0, 0]:.4f}")

    # Check that min_lfsr equals the minimum
    expected_min_lfsr = jnp.min(lfsr, axis=0)
    assert jnp.allclose(min_lfsr, expected_min_lfsr), "min_lfsr doesn't match expected"
    print(f"\n✓ PASSED: min-lfsr correctly computes minimum across L effects")


def test_lfsr_from_scfm():
    """Test LFSR computation using the actual SCFM implementation."""
    print("\n" + "=" * 60)
    print("Testing: LFSR from SCFM infer module")
    print("=" * 60)

    from scfm.infer import _compute_clfsr
    from scfm.params import PosteriorParams

    key = rdm.PRNGKey(789)

    # Test parameters
    L = 2
    p = 10
    k = 3
    n_samples = 100000

    key, subkey1, subkey2, subkey3 = rdm.split(key, 4)

    # Generate random posterior parameters
    alpha = rdm.uniform(subkey1, shape=(L, p))
    mean_b = rdm.normal(subkey2, shape=(L, p, k)) * 0.5

    var_b = jnp.zeros((L, p, k, k))
    for l in range(L):
        for j in range(p):
            key, subkey = rdm.split(key)
            A = rdm.normal(subkey, shape=(k, k)) * 0.3
            cov = A @ A.T + 0.1 * jnp.eye(k)
            var_b = var_b.at[l, j].set(cov)

    # Create PosteriorParams
    post = PosteriorParams(prob=alpha, mean_b=mean_b, var_b=var_b)

    # Compute SCFM's clfsr
    scfm_clfsr = _compute_clfsr(post)

    # Compute our reference analytical clfsr
    ref_clfsr = compute_analytical_clfsr(mean_b, var_b)

    # Compare
    diff = jnp.abs(scfm_clfsr - ref_clfsr)
    max_diff = jnp.max(diff)

    print(f"\nSCFM _compute_clfsr vs reference analytical clfsr:")
    print(f"  Max absolute difference: {max_diff:.10f}")

    assert jnp.allclose(scfm_clfsr, ref_clfsr), "SCFM clfsr doesn't match reference"
    print(f"✓ PASSED: SCFM _compute_clfsr matches analytical formula exactly")

    # Now test the full LFSR computation (as done in finemap)
    # min_lfsr = min_l(1 - α * (1 - clfsr))
    scfm_min_lfsr = jnp.min(1.0 - alpha[:, :, jnp.newaxis] * (1.0 - scfm_clfsr), axis=0)

    # Compare to empirical
    empirical_lfsr = sample_empirical_lfsr(alpha, mean_b, var_b, n_samples, subkey3)
    empirical_min_lfsr = jnp.min(empirical_lfsr, axis=0)

    diff = jnp.abs(scfm_min_lfsr - empirical_min_lfsr)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"\nSCFM min-lfsr vs empirical sampling ({n_samples:,} samples):")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    tolerance = 0.02
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"
    print(f"✓ PASSED: SCFM min-lfsr matches empirical within {tolerance:.1%} tolerance")


def run_all_tests():
    """Run all LFSR validation tests."""
    print("\n" + "=" * 70)
    print("LFSR (Local False Sign Rate) Empirical Validation Tests")
    print("=" * 70)

    test_clfsr_matches_empirical()
    test_lfsr_matches_empirical()
    test_min_lfsr()
    test_lfsr_from_scfm()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
