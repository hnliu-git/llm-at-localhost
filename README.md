[![Packaged with Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Coverage](res/coverage.svg)

# LLM@localhost

This is a template for my Python Project. Poetry is used for dependencies management and building. Pytest is used for testing, flake8 is used as a passive linter and black as an aggressive linter.

## MakeFile
By default, we will use the same python when you run `python` in your environment.

In [Makefile](Makefile), you will find some available actions.  For example
```
make install
```
Let you setup all the dependencies
```
make test
```
Run all the tests and create coverage badge

## Workflow

In [python-app.yml](.github/workflows/python-app.yml), you see a basic workflow for setting up the project and run the test. This is for Continous Integreation purpose.