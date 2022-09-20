# -*- coding: utf-8 -*-

import re

from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.install import install as _install

VERSION_PY = """
# This file is originally generated from Git information by running 'setup.py
# version'. Distribution tarballs contain a pre-generated copy of this file.

__version__ = '%s'
"""


def get_version():
    try:
        f = open("deeptools/_version.py")
    except EnvironmentError:
        return None
    for line in f.readlines():
        mo = re.match("__version__ = '([^']+)'", line)
        if mo:
            ver = mo.group(1)
            return ver
    return None


class sdist(_sdist):

    def run(self):
        self.distribution.metadata.version = get_version()
        return _sdist.run(self)


class install(_install):

    def run(self):
        self.distribution.metadata.version = get_version()
        _install.run(self)
        return


def openREADME():
    """
    This is only needed because README.rst is UTF-8 encoded and that won't work
    under python3 iff sys.getfilesystemencoding() returns 'ascii'

    Since open() doesn't accept an encoding in python2...
    """
    try:
        f = open("README.md", encoding="utf-8")
    except:
        f = open("README.md")

    foo = f.read()
    f.close()
    return foo


setup(
    name='cfDNA_GCcorrection',
    version=get_version(),
    author='Sebastian Röner',
    author_email='sebastian.roener@bih-charite.de',
    packages=find_packages(),
    scripts=['bin/computeGCBias_readlen','bin/correctGCBias_readlen'],
    include_package_data=True,
    url='http://pypi.python.org/pypi/cfDNA_GCcorrection/',
    license='LICENSE.txt',
    description='Tools for correcting GC bias in cell-free DNA samples',
    long_description=openREADME(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
    install_requires=[
        "pandas",
        "numpy >= 1.9.0",
        "scipy >= 0.17.0",
        "matplotlib >= 3.3.0",
        "pysam >= 0.14.0",
        "numpydoc >= 0.5",
        "pyBigWig >= 0.2.1",
        "py2bit >= 0.2.0",
        "plotly >= 4.9",
        "deeptoolsintervals >= 0.1.8",
        "csaps"
    ],
    zip_safe=True,
    cmdclass={'sdist': sdist, 'install': install}
)
