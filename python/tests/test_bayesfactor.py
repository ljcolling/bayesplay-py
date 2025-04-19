from math import sqrt

from bayesplay_py import Likelihood, Prior
from pytest import approx


def test_default_t_test():
    t = 2.03
    n = 80

    d = t / sqrt(n)

    # Define the likelihood using a d scaled non-central t-distribution
    likelihood = Likelihood.noncentral_d(d, n)

    # Define the priors

    # A Cauchy prior for the alternative hypothesis
    h1_prior = Prior.cauchy(location=0, scale=1)

    # A point null prior
    h0_prior = Prior.point(point=0)

    # Build your models by multiplying the likelihood by the priors
    m1 = likelihood * h1_prior
    m0 = likelihood * h0_prior

    # Integrate the models to get the evidence
    m1_evidence = m1.integrate()
    m0_evidence = m0.integrate()

    # Compute a Bayes factor by comparison the model evidence
    bf = m1_evidence / m0_evidence
    assert bf == approx(0.6420764880775981)

    # Compute the Savage-Dickey ratio
    posterior = m1.posterior
    sd_bf = h1_prior(0) / posterior(0)

    assert sd_bf == bf
