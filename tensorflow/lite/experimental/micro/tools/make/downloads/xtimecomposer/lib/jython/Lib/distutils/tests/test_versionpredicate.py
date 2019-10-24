"""Tests harness for distutils.versionpredicate.

"""

import distutils.versionpredicate
import doctest

def test_suite():
    return doctest.DocTestSuite(distutils.versionpredicate)
