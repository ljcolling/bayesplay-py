from math import sqrt

from bayesplay_py import Evidence, Likelihood, Model, Prior
from pytest import approx


def test_compute():
    t = 2.03
    n = 80

    d = t / sqrt(n)

    # Define the likelihood using a d scaled non-central t-distribution
    likelihood: Prior = Likelihood.noncentral_d(d, n)

    # Define the priors

    # A Cauchy prior for the alternative hypothesis
    h1_prior: Prior = Prior.cauchy(location=0, scale=1)

    # A point null prior
    h0_prior: Prior = Prior.point(point=0)

    # Build your models by multiplying the likelihood by the priors
    m1: Model = likelihood * h1_prior
    m0: Model = likelihood * h0_prior

    # Integrate the models to get the evidence
    m1_evidence: Evidence = m1.integrate()
    m0_evidence: Evidence = m0.integrate()

    # Compute a Bayes factor by comparison the model evidence
    bf = m1_evidence / m0_evidence
    assert bf == approx(0.6420764880775981)
