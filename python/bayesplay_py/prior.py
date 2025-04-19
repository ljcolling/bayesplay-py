from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, overload

from pydantic import BaseModel

import bayesplay_py._lib as _lib
from .common import Param, ParamList
from .model import Model

if TYPE_CHECKING:
    from bayesplay_py._lib import PythonPrior
    from .likelihood import Likelihood


class PriorFamily(str, Enum):
    normal = "normal"
    cauchy = "cauchy"
    point = "point"
    student_t = "student_t"
    beta = "beta"


class PriorInterface(BaseModel):
    family: PriorFamily
    params: ParamList


class Prior:
    _family: PriorFamily
    _params: ParamList
    _object: PythonPrior | None = None
    _interface: PriorInterface
    _initialisation_func: Callable[[dict[str, Any]], PythonPrior] = _lib.init_likelihood

    _initialisation_func = _lib.init_prior

    def __init__(self, family: PriorFamily, **kwargs: float | None):
        self._family = family
        params = {n: v for n, v in kwargs.items() if v is not None}
        self._params = ParamList(
            [
                Param(name=name, value=value)
                for name, value in params.items()
                if value is not None
            ]
        )

        self._obj = None
        self._interface: PriorInterface = PriorInterface(
            family=self._family, params=self._params
        )

    def initialise_object(self):
        if self._object is None:
            model_dump = self._interface.model_dump()
            self._object = self._initialisation_func(model_dump)

    @staticmethod
    def normal(
        mean: float, se: float, ll: float | None = None, ul: float | None = None
    ) -> Prior:
        return Prior(family=PriorFamily.normal, mean=mean, sd=se, ll=ll, ul=ul)

    @staticmethod
    def cauchy(
        location: float, scale: float, ll: float | None = None, ul: float | None = None
    ) -> Prior:
        return Prior(
            family=PriorFamily.cauchy, location=location, scale=scale, ll=ll, ul=ul
        )

    @staticmethod
    def point(point: float) -> Prior:
        return Prior(family=PriorFamily.point, point=point)

    @staticmethod
    def student_t(
        mean: float,
        sd: float,
        df: float,
        ll: float | None = None,
        ul: float | None = None,
    ) -> Prior:
        return Prior(family=PriorFamily.student_t, mean=mean, sd=sd, df=df)

    @staticmethod
    def beta(
        alpha: float, beta: float, ll: float | None = None, ul: float | None = None
    ) -> Prior:
        return Prior(family=PriorFamily.beta, alpha=alpha, beta=beta, ll=ll, ul=ul)

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: list[float]) -> list[float]: ...
    def __call__(self, x: float | list[float]) -> float | list[float]:
        if self._obj is None:
            self._obj = _lib.init_prior(self._interface.model_dump())
        return self.function(x)

    def integrate(self, lb: float | None = None, ub: float | None = None) -> float:
        if self._obj is None:
            self._obj = _lib.init_prior(self._interface.model_dump())
        return self._obj.integrate(lb, ub)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        if self._obj is None:
            self._obj = _lib.init_prior(self._interface.model_dump())
        if isinstance(x, list):
            return self._obj.function_vec(x)
        else:
            return self._obj.function(x)

    def __mul__(self, other: Likelihood):
        return Model(other, self)
