from bayesplay_py import Prior


def test_prior_normal():
    prior = Prior.normal(mean=0, sd=1)
    assert prior._family == "normal"
    assert prior._params.get("mean") == 0
    assert prior._params.get("sd") == 1

def test_prior_cauchy():
    prior = Prior.cauchy(location=0, scale=1)
    assert prior._family == "cauchy"
    assert prior._params.get("location") == 0
    assert prior._params.get("scale") == 1

def test_prior_point():
    prior = Prior.point(point=0)
    assert prior._family == "point"
    assert prior._params.get("point") == 0



def test_prior_integrate_normal():
    prior = Prior.normal(mean=0, sd=1)
    result = prior.integrate(-float("inf"), float("inf"))
    assert abs(result - 1.0) < 1e-6

def test_prior_integrate_partial():
    prior = Prior.normal(mean=0, sd=1)
    result = prior.integrate(-1, 1)
    assert abs(result - 0.6827) < 0.01

def test_prior_integrate_point():
    prior = Prior.point(point=2)
    assert prior.integrate(1.5, 2.5) == 1.0
    assert prior.integrate(2.1, 3.0) == 0.0


def test_prior_function_normal():
    prior = Prior.normal(mean=0, sd=1)
    # Standard normal at 0 is about 0.3989
    assert abs(prior(0) - 0.3989) < 1e-3
    # Symmetry: f(-1) == f(1)
    assert abs(prior(-1) - prior(1)) < 1e-8

def test_prior_function_point():
    prior = Prior.point(point=3)
    assert prior(3) == 1
    assert prior(2.9) == 0.0
    assert prior(3.1) == 0.0


def test_prior_student_t():
    prior = Prior.student_t(mean=0, sd=1, df=3)
    assert prior._family == "student_t"
    assert prior._params.get("mean") == 0
    assert prior._params.get("sd") == 1
    assert prior._params.get("df") == 3

def test_prior_beta():
    prior = Prior.beta(alpha=2, beta=5)
    assert prior._family == "beta"
    assert prior._params.get("alpha") == 2
    assert prior._params.get("beta") == 5


def test_prior_student_t_function():
    prior = Prior.student_t(mean=0, sd=1, df=3)
    # PDF at mean
    pdf_at_0 = prior.function(0)
    assert isinstance(pdf_at_0, float)
    assert pdf_at_0 > 0

def test_prior_student_t_integrate():
    prior = Prior.student_t(mean=0, sd=1, df=3)
    # Integrate over symmetric interval
    integral = prior.integrate(-1, 1)
    assert 0 < integral < 1

def test_prior_beta_function():
    prior = Prior.beta(alpha=2, beta=5)
    # PDF at 0.5
    pdf_at_05 = prior.function(0.5)
    assert isinstance(pdf_at_05, float)
    assert pdf_at_05 > 0

def test_prior_beta_integrate():
    prior = Prior.beta(alpha=2, beta=5)
    # Integrate over [0, 1]
    integral = prior.integrate(0, 1)
    assert abs(integral - 1.0) < 1e-6
