"""Custom op used by periodic_intersperse."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.periodic_intersperse.python import *
# pylint: enable=unused-import,wildcard-import

periodic_intersperse =\
    periodic_intersperse_op._periodic_intersperse_op.periodic_intersperse
