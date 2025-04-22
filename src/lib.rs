use anyhow::anyhow;
use bayesplay::prelude::{Function, Integrate, Likelihood, Model, Posterior, Prior};
use interface::{
    InterfaceError, LikelihoodFamily, LikelihoodInterface, ParamDefinition, ParamSetting,
    ParameterName, PriorFamily, PriorInterface,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use pyo3::types::PyDict;

#[pyclass]
struct PythonLikelihood {
    inner: Likelihood,
}

#[pymethods]
impl PythonLikelihood {
    fn function(&self, x: f64) -> PyResult<f64> {
        self.inner
            .function(x)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn function_vec(&self, x: Vec<f64>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .function(x.as_slice())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PythonPrior {
    inner: Prior,
}

#[pymethods]
impl PythonPrior {
    fn function(&self, x: f64) -> PyResult<f64> {
        self.inner
            .function(x)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn function_vec(&self, x: Vec<f64>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .function(x.as_slice())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> PyResult<f64> {
        self.inner
            .integrate(lb, ub)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PythonModel {
    inner: Model,
}

#[pymethods]
impl PythonModel {
    fn integral(&self) -> PyResult<f64> {
        self.inner
            .integral()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PythonPosterior {
    inner: Posterior,
}

#[pymethods]
impl PythonPosterior {
    fn function(&self, x: f64) -> PyResult<f64> {
        self.inner
            .function(x)
            .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))
    }

    fn function_vec(&self, x: Vec<f64>) -> PyResult<Vec<Option<f64>>> {
        self.inner
            .function(x.as_slice())
            .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))
    }

    fn integrate(&self, lb: Option<f64>, ub: Option<f64>) -> PyResult<f64> {
        self.inner
            .integrate(lb, ub)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

struct PyInterface<'a>(Option<&'a Bound<'a, PyDict>>);
struct RustInterface {
    family: String,
    params: ParamDefinition,
}

impl TryFrom<PyInterface<'_>> for RustInterface {
    type Error = anyhow::Error;
    fn try_from(val: PyInterface<'_>) -> anyhow::Result<RustInterface> {
        let obj = val.0.ok_or(anyhow!("Could not parse input"))?;

        let family: String = obj
            .get_item("family")?
            .ok_or(anyhow!("Could not parse `family`"))?
            .extract()?;

        let params = obj
            .get_item("params")?
            .ok_or(anyhow!("Could not parse `params`"))?;

        let mut param_settings = Vec::with_capacity(params.len()?);
        for i in 0..params.len()? {
            let param = params.get_item(i)?;
            let name = ParameterName::try_from(param.get_item("name")?.extract::<String>()?)
                .map_err(|e| anyhow!(e))?;

            let value = Some(param.get_item("value")?.extract::<f64>()?);

            param_settings.push(ParamSetting { name, value });
        }
        let params = ParamDefinition(param_settings);

        Ok(RustInterface { family, params })
    }
}

impl TryFrom<RustInterface> for LikelihoodInterface {
    type Error = anyhow::Error;
    fn try_from(value: RustInterface) -> anyhow::Result<LikelihoodInterface> {
        let family: LikelihoodFamily =
            LikelihoodFamily::try_from(value.family.as_str()).map_err(|e| anyhow!(e))?;

        let params = value.params;
        Ok(LikelihoodInterface { family, params })
    }
}

impl TryFrom<RustInterface> for PriorInterface {
    type Error = anyhow::Error;
    fn try_from(value: RustInterface) -> anyhow::Result<PriorInterface> {
        let family: PriorFamily =
            PriorFamily::try_from(value.family.as_str()).map_err(|e| anyhow!(e))?;

        let params = value.params;
        Ok(PriorInterface { family, params })
    }
}

#[pyfunction]
fn init_likelihood(likelhood: Option<&Bound<'_, PyDict>>) -> PyResult<PythonLikelihood> {
    let rust_interface = RustInterface::try_from(PyInterface(likelhood))
        .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;

    let likelihood_interface = LikelihoodInterface::try_from(rust_interface)
        .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;

    let likelihood = Likelihood::try_from(likelihood_interface)
        .map_err(|e: InterfaceError| PyValueError::new_err(e.to_string()))?;

    Ok(PythonLikelihood { inner: likelihood })
}

#[pyfunction]
fn init_prior(prior: Option<&Bound<'_, PyDict>>) -> PyResult<PythonPrior> {
    let rust_interface = RustInterface::try_from(PyInterface(prior))
        .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;

    let prior_interface = PriorInterface::try_from(rust_interface)
        .map_err(|e: anyhow::Error| PyValueError::new_err(e.to_string()))?;

    let prior = Prior::try_from(prior_interface)
        .map_err(|e: InterfaceError| PyValueError::new_err(e.to_string()))?;

    Ok(PythonPrior { inner: prior })
}

#[pyfunction]
fn init_model(
    likelihood: Option<&Bound<'_, PyDict>>,
    prior: Option<&Bound<'_, PyDict>>,
) -> PyResult<PythonModel> {
    let prior = init_prior(prior)?.inner;
    let likelihood = init_likelihood(likelihood)?.inner;

    let model: Model = prior * likelihood;
    Ok(PythonModel { inner: model })
}

#[pyfunction]
fn init_posterior(
    likelihood: Option<&Bound<'_, PyDict>>,
    prior: Option<&Bound<'_, PyDict>>,
) -> PyResult<PythonPosterior> {
    let prior = init_prior(prior)?.inner;
    let likelihood = init_likelihood(likelihood)?.inner;

    let model: Model = prior * likelihood;
    let posterior = model
        .posterior()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PythonPosterior { inner: posterior })
}
/// A Python module implemented in Rust.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(init_prior, m)?)?;
    m.add_function(wrap_pyfunction!(init_model, m)?)?;
    m.add_function(wrap_pyfunction!(init_posterior, m)?)?;
    m.add_class::<PythonLikelihood>()?;
    m.add_class::<PythonPrior>()?;
    m.add_class::<PythonModel>()?;
    m.add_class::<PythonPosterior>()?;
    Ok(())
}
