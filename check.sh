#!/bin/bash
autoflake --in-place --remove-unused-variables -r --remove-all-unused-imports .
mypy --install-types
pre-commit run --all-files