from collections import UserList

from pydantic import BaseModel, model_serializer
from pydantic.dataclasses import dataclass


class Definition:
    def initialise_object(self):
        if self._object is None:
            self._object = self._initialisation_func(self._interface.model_dump())

class Param(BaseModel):
    name: str
    value: float


@dataclass
class ParamList(UserList[Param]):
    data: list[Param]

    def get(self, name: str) -> float:
        return {item.name: item.value for item in self}.get(name)

    @model_serializer
    def serialize_model(self) -> list[Param]:
        return self.data

    def __repr__(self) -> str:
        return self.data.__repr__()
