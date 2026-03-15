import numpy as np




def top1_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    return np.max(p)

def top1_minus_top2_mass(probs):
    p = np.asarray(probs, dtype=np.float64)
    p_sorted = np.sort(p)[::-1]
    if len(p_sorted) < 2:
        return np.nan
    return p_sorted[0] - p_sorted[1]




def js_between_uniform(probs, eps=1e-12, log_base=np.e):
    """
    Jensen–Shannon divergence between a categorical distribution and uniform.

    Args:
        probs: iterable of non-negative numbers
        eps: small constant for numerical stability
        log_base: np.e (nats) or 2 (bits)

    Returns:
        JSD(P || Uniform)
    """
    p = np.asarray(probs, dtype=np.float64)
    total = p.sum()

    if total <= 0:
        return np.nan

    # normalize
    p = p / total
    K = len(p)
    u = np.full(K, 1.0 / K)

    # mixture
    m = 0.5 * (p + u)

    # avoid log(0)
    p = np.clip(p, eps, 1.0)
    m = np.clip(m, eps, 1.0)

    # KL(P || M)
    kl_pm = np.sum(p * np.log(p / m))

    # KL(U || M)
    kl_um = np.sum(u * np.log(u / m))

    js = 0.5 * (kl_pm + kl_um)

    # change log base if needed
    if log_base != np.e:
        js /= np.log(log_base)

    return js


