PKGDIR  := vetk
TESTDIR := test
PYTHON  := /opt/anaconda3/bin/python3

.PHONY: help wheel build clean

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  build       to make package distribution"
	@echo "  wheel       to make wheel binary distribution"
	@echo "  clean       to remove build, cache, and temporary files"

build:
	$(PYTHON) setup.py build

wheel:
	$(PYTHON) setup.py bdist_wheel

clean:
	rm -rf flake8.out
	rm -rf *.egg-info .eggs
	rm -rf .tox htmlcov
	rm -rf "$(PKGDIR)/__pycache__"
	rm -rf "$(TESTDIR)/__pycache__"
