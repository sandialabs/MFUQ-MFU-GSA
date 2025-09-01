## Test suite dependencies
You'll need to install `pytest` and `pytest-cov` to run tests and generate code coverage reports.
These can be installed using `pip`. 

## How to add a unit test

We are using the pytest unit test module.  
Your test file should begin with "test" such as `test_ADE.py`.

Be sure to import pytest and then to set the relative path to where the python files are found. 
This will allow you to import files such as generalizedADE.py.

```
import pytest
import os, sys

sys.path.insert(0,'../')

from generalizedADE import generalizedADE, FRADE
```

Unit tests should be functions whose names begin with "test". 
For example:

```
def test_FRADE():
    params = { "nu": 0.1, "alpha": 2.0 }
    model = FRADE()
    model.solve(params)
    
    # Add a check with an analytical solution.
```

## How to run the tests

### Quick command summary
```
pytest --disable-warnings
pytest --disable-warnings --cov ../python/network_tool --cov-report html
git clean -df
```
The first command would run pytest without creating any code coverage report.
The second command would create an HTML code coverage report.
The third command deletes everything in the test folder that is not tracked in the git repository. This enables easy cleanup
for any files generated during testing.

The script `run_tests.sh` will run the second and third commands.

### Running the tests
- First change (cd) to the top level "tests" directory
- Run all tests using `run_tests.sh`
```
