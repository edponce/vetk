PKGDIR  := vetk
TESTDIR := tests
PYTHON  := /opt/anaconda3/bin/python3
DOCDIR  := docs

.PHONY: help wheel build clean docs

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
	rm -rf .tox
	rm -rf .coverage *coverage.* htmlcov
	rm -rf "$(PKGDIR)"/__pycache__
	rm -rf "$(PKGDIR)"/clustering/__pycache__
	rm -rf "$(TESTDIR)"/__pycache__
	$(MAKE) -C "$(DOCDIR)" clean

docs:
	$(MAKE) -C "$(DOCDIR)" html
