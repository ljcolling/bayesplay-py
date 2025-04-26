from dataclasses import dataclass
from typing import Any, Dict, Generic, List, TypeVar


@dataclass
class Param:
    name: str
    value: float


@dataclass
class ParamList:
    data: List[Param]

    def get(self, name: str) -> float:
        params = {item.name: item.value for item in self.data}
        if name not in params:
            raise ValueError(f"Parameter {name} not found in parameter list.")
        return params[name]

    def __repr__(self) -> str:
        return self.data.__repr__()



T = TypeVar("T")

@dataclass
class Interface(Generic[T]):
    family: T
    params: ParamList

    def model_dump(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "params": [{"name": v.name, "value": v.value} for v in self.params.data],
        }
