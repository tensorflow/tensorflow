#!/usr/bin/env python

from setuptools import setup, find_packages

META_DATA = dict(
    name = 'skflow',
    version = '0.0.1',
    url = 'https://github.com/google/skflow',
    license = 'Apache-2',
    packages = find_packages(),
    install_requires = [
        'sklearn',
        'scipy',
        'numpy',
    ],
)


if __name__ == '__main__':
    setup(**META_DATA)

