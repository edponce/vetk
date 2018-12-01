#!/usr/bin/env python3

from setuptools import setup, find_packages
import vetk as pkg


# Load long description from files
try:
    with open('README.rst') as fd:
        long_description = fd.read()
except IOError:
    long_description = ''

try:
    with open('HISTORY.rst') as fd:
        if long_description:
            long_description += '\n\n'
        long_description += fd.read()
except IOError:
    pass

# A list of strings specifying what other distributions need to be installed
# when this package is installed.
install_requirements = [
    'numpy>=1.14',
    'matplotlib>=2.2',
    'scikit-learn>=0.19',
    'scipy>=1.1'
]

# A list of strings specifying what other distributions need to be present
# in order for this setup script to run.
setup_requirements = [
    'setuptools>=39.1',
    'wheel>=0.31'
]

# A list of strings specifying what other distributions need to be present
# for this package tests to run.
tests_requirements = [
    'tox>=3.5',
    'coverage>=4.5'
]

# A dictionary mapping of names of "extra" features to lists of strings
# describing those features' requirements. These requirements will not be
# automatically installed unless another package depends on them.
extras_requirements = {
    'lint': ['flake8>=3.6'],
    'reST': ['Sphinx>=1.7']
}

# For PyPI, the 'download_url' is a link to a hosted repository.
# Github hosting creates tarballs for download at
#   https://github.com/{username}/{package}/archive/{tag}.tar.gz.
# To create a git tag
#   git tag pkg.__name__-pkg.__version__ -m 'Adds a tag so that we can put package on PyPI'
#   git push --tags origin master
setup(
    name = pkg.__name__,
    version = pkg.__version__,
    description = pkg.__description__,
    long_description = long_description,
    keywords = pkg.__keywords__,
    url = pkg.__url__,
    download_url = '{}/archive/{}-{}.tar.gz'.format(pkg.__url__,
                                                    pkg.__name__,
                                                    pkg.__version__),
    author = pkg.__author__,
    author_email = pkg.__author_email__,
    license = pkg.__license__,
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis'
        ],
    platforms = ['Linux'],
    zip_safe = False,
    python_requires = '>=3',
    include_package_data = True,
    packages=find_packages(),
    install_requires = install_requirements,
    setup_requires = setup_requirements,
    extras_require = extras_requirements,
    tests_require = tests_requirements,
    test_suite = 'test'
)
