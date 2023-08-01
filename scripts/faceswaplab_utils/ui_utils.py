from dataclasses import fields, is_dataclass
from typing import *


def dataclass_from_flat_list(cls: type, values: Tuple[Any, ...]) -> Any:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    idx = 0
    init_values = {}
    for field in fields(cls):
        if is_dataclass(field.type):
            inner_values = [values[idx + i] for i in range(len(fields(field.type)))]
            init_values[field.name] = field.type(*inner_values)
            idx += len(inner_values)
        else:
            value = values[idx]
            init_values[field.name] = value
            idx += 1
    return cls(**init_values)


def dataclasses_from_flat_list(
    classes_mapping: List[type], values: Tuple[Any, ...]
) -> List[Any]:
    instances = []
    idx = 0
    for cls in classes_mapping:
        num_fields = sum(
            len(fields(field.type)) if is_dataclass(field.type) else 1
            for field in fields(cls)
        )
        instance = dataclass_from_flat_list(cls, values[idx : idx + num_fields])
        instances.append(instance)
        idx += num_fields
    assert [
        isinstance(i, t) for i, t in zip(instances, classes_mapping)
    ], "Instances should match types"
    return instances
