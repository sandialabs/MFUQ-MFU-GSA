#!/bin/bash

# The -s flag is related to output capture. -s means not to capture stdout or stderr.
# More on this here: https://docs.pytest.org/en/7.1.x/how-to/capture-stdout-stderr.html
pytest -s --disable-warnings --cov ../ --cov-report html && git clean -df
