from typing import Any
from modules.shared import opts


def get_sd_option(name: str, default: Any) -> Any:
    assert opts.data is not None
    return opts.data.get(name, default)
