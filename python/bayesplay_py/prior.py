from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, overload

import bayesplay_py._lib as _lib
from .common import Interface, Param, ParamList

if TYPE_CHECKING:
    from ._lib import PythonPrior
    from .likelihood import Likelihood


class PriorFamily(str, Enum):
    normal = "normal"
    cauchy = "cauchy"
    point = "point"
    student_t = "student_t"
    beta = "beta"


@dataclass
class PriorInterface(Interface[PriorFamily]):
    family: PriorFamily
    params: ParamList


class Prior:
    _family: PriorFamily
    _params: ParamList
    _object: PythonPrior
    _initialisation_func: Callable[[Dict[str, Any]], PythonPrior] = _lib.init_prior

    def __init__(self, family: PriorFamily, **kwargs: Optional[float]):
        self._family = family
        params = {n: v for n, v in kwargs.items() if v is not None}
        self._params = ParamList(
            [Param(name=name, value=value) for name, value in params.items()]
        )

        self._interface: PriorInterface = PriorInterface(
            family=self._family, params=self._params
        )
        model_dump: dict[str, Any] = self._interface.model_dump()
        self._object = self._initialisation_func(model_dump)

    @classmethod
    def normal(
        cls,
        mean: float,
        sd: float,
        ll: Optional[float] = None,
        ul: Optional[float] = None,
    ):
        return cls(family=PriorFamily.normal, mean=mean, sd=sd, ll=ll, ul=ul)

    @classmethod
    def cauchy(
        cls,
        location: float,
        scale: float,
        ll: Optional[float] = None,
        ul: Optional[float] = None,
    ):
        return cls(
            family=PriorFamily.cauchy, location=location, scale=scale, ll=ll, ul=ul
        )

    @classmethod
    def point(cls, point: float):
        return cls(family=PriorFamily.point, point=point)

    @classmethod
    def student_t(
        cls,
        mean: float,
        sd: float,
        df: float,
        ll: Optional[float] = None,
        ul: Optional[float] = None,
    ):
        return cls(family=PriorFamily.student_t, mean=mean, sd=sd, df=df, ll=ll, ul=ul)

    @classmethod
    def beta(
        cls,
        alpha: float,
        beta: float,
        ll: Optional[float] = None,
        ul: Optional[float] = None,
    ):
        return cls(family=PriorFamily.beta, alpha=alpha, beta=beta, ll=ll, ul=ul)

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: List[float]) -> List[float]: ...
    def __call__(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
        return self.function(x)

    def integrate(
        self, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> float:
        return self._object.integrate(lb, ub)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: List[float]) -> List[float]: ...
    def function(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
        if isinstance(x, List):
            return self._object.function_vec(x)
        else:
            return self._object.function(x)

    def __mul__(self, other: Likelihood):
        from .model import Model

        return Model(other, self)
