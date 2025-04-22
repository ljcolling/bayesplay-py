from bayesplay_py import Likelihood


def test_likelihood_normal():
    likelihood = Likelihood.normal(mean=0, se=1)
    assert likelihood._family == "normal"
    assert likelihood._params.get("mean") == 0
    assert likelihood._params.get("se") == 1


def test_likelihood_noncentral_d():
    likelihood = Likelihood.noncentral_d(d=0.5, n=10)
    assert likelihood._family == "noncentral_d"
    assert likelihood._params.get("d") == 0.5
    assert likelihood._params.get("n") == 10


def test_likelihood_noncentral_d2():
    likelihood = Likelihood.noncentral_d2(d=0.5, n1=20, n2=25)
    assert likelihood._family == "noncentral_d2"
    assert likelihood._params.get("d") == 0.5
    assert likelihood._params.get("n1") == 20
    assert likelihood._params.get("n2") == 25


def test_likelihood_noncentral_t():
    likelihood = Likelihood.noncentral_t(t=2.0, df=10)
    assert likelihood._family == "noncentral_t"
    assert likelihood._params.get("t") == 2.0
    assert likelihood._params.get("df") == 10


def test_likelihood_student_t():
    likelihood = Likelihood.student_t(mean=0, sd=1, df=5)
    assert likelihood._family == "student_t"
    assert likelihood._params.get("mean") == 0
    assert likelihood._params.get("sd") == 1
    assert likelihood._params.get("df") == 5


def test_likelihood_binomial():
    likelihood = Likelihood.binomial(successes=3, trials=10)
    assert likelihood._family == "binomial"
    assert likelihood._params.get("successes") == 3
    assert likelihood._params.get("trials") == 10


def test_likelihood_function_noncentral_d():
    likelihood = Likelihood.noncentral_d(d=0.5, n=30)
    assert abs(likelihood.function(0.5) - likelihood(0.5)) < 1e-12
    vals = likelihood.function([0.4, 0.6])
    vals_call = likelihood([0.4, 0.6])
    assert all(abs(a - b) < 1e-12 for a, b in zip(vals, vals_call))

def test_likelihood_function_noncentral_d2():
    likelihood = Likelihood.noncentral_d2(d=0.5, n1=20, n2=25)
    assert abs(likelihood.function(0.5) - likelihood(0.5)) < 1e-12
    vals = likelihood.function([0.4, 0.6])
    vals_call = likelihood([0.4, 0.6])
    assert all(abs(a - b) < 1e-12 for a, b in zip(vals, vals_call))

def test_likelihood_function_noncentral_t():
    likelihood = Likelihood.noncentral_t(t=2.0, df=1.5)
    assert abs(likelihood.function(2.0) - likelihood(2.0)) < 1e-12
    vals = likelihood.function([1.5, 2.5])
    vals_call = likelihood([1.5, 2.5])
    assert all(abs(a - b) < 1e-12 for a, b in zip(vals, vals_call))

def test_likelihood_function_student_t():
    likelihood = Likelihood.student_t(mean=0, sd=1, df=5)
    assert abs(likelihood.function(2.0) - likelihood(2.0)) < 1e-12
    vals = likelihood.function([1.5, 2.5])
    vals_call = likelihood([1.5, 2.5])
    assert all(abs(a - b) < 1e-12 for a, b in zip(vals, vals_call))

def test_likelihood_function_binomial():
    likelihood = Likelihood.binomial(successes=5, trials=10)
    assert abs(likelihood.function(0.5) - likelihood(0.5)) < 1e-12
    vals = likelihood.function([0.4, 0.6])
    vals_call = likelihood([0.4, 0.6])
    assert all(abs(a - b) < 1e-12 for a, b in zip(vals, vals_call))
