import numpy as np
from scipy.linalg import sqrtm, eigvals

def density_matrix_from_returns(returns):
    """
    Build a density matrix (quantum state) from ETF log returns.
    Use the covariance matrix normalized to trace 1.
    """
    cov = np.cov(returns.T, ddof=1)
    # Add small diagonal to ensure positive definiteness
    cov += 1e-8 * np.eye(cov.shape[0])
    # Normalize to trace 1
    rho = cov / np.trace(cov)
    return rho

def petz_map(prior, likelihood, tau=0.5):
    """
    Petz map quantum Bayes rule: posterior = L^{1-tau} (prior) L^{-tau} etc.
    Simplified: posterior = (prior^{tau} @ likelihood^{1-tau} @ prior^{tau}) / trace
    """
    # Matrix powers
    prior_pow = np.linalg.matrix_power(prior, tau)
    likelihood_pow = np.linalg.matrix_power(likelihood, 1-tau)
    posterior = prior_pow @ likelihood_pow @ prior_pow
    posterior = (posterior + posterior.T) / 2  # symmetrize
    posterior = posterior / np.trace(posterior)
    return posterior

def fidelity_based_update(prior, likelihood, tau=0.5):
    """
    Heuristic: posterior = (1-tau)*prior + tau*likelihood (geometric mean mixing)
    """
    posterior = (1-tau) * prior + tau * likelihood
    posterior = posterior / np.trace(posterior)
    return posterior

def bayesian_scores(returns, tau=0.5, use_petz=True):
    """
    Compute per‑ETF Bayesian update score = trace distance between prior and posterior.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    # Split window into two halves: prior (first half) and likelihood (second half)
    split = len(returns_clean) // 2
    prior_returns = returns_clean.iloc[:split]
    likelihood_returns = returns_clean.iloc[split:]
    if prior_returns.shape[0] < 2 or likelihood_returns.shape[0] < 2:
        return {t: 0.0 for t in returns_clean.columns}
    rho_prior = density_matrix_from_returns(prior_returns)
    rho_likelihood = density_matrix_from_returns(likelihood_returns)
    if use_petz:
        rho_posterior = petz_map(rho_prior, rho_likelihood, tau)
    else:
        rho_posterior = fidelity_based_update(rho_prior, rho_likelihood, tau)
    # Trace distance: 0.5 * sum |λ_i|
    diff = rho_posterior - rho_prior
    ev = eigvals(diff)
    trace_dist = 0.5 * np.sum(np.abs(ev))
    # Per‑ETF score: how much each diagonal element changed? Use diagonal relative change
    prior_diag = np.diag(rho_prior)
    post_diag = np.diag(rho_posterior)
    delta = np.abs(post_diag - prior_diag)
    tickers = returns_clean.columns
    return {ticker: delta[i] for i, ticker in enumerate(tickers)}
