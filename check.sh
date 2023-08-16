#!/bin/bash
autoflake --in-place --remove-unused-variables -r --remove-all-unused-imports .
mypy --non-interactive --install-types
pre-commit run --all-files
