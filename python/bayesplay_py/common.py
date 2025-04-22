from collections import UserList
from typing import List

from pydantic import BaseModel, model_serializer
from pydantic.dataclasses import dataclass


class Param(BaseModel):
    name: str
    value: float


@dataclass
class ParamList(UserList[Param]):
    data: List[Param]

    def get(self, name: str) -> float:
        params = {item.name: item.value for item in self}
        if name not in params:
            raise ValueError(f"Parameter {name} not found in parameter list.")
        return params[name]

    @model_serializer
    def serialize_model(self) -> List[Param]:
        return self.data

    def __repr__(self) -> str:
        return self.data.__repr__()
