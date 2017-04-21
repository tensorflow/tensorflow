"""Custom op used by shuffle."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.shuffle.python import *
# pylint: enable=unused-import,wildcard-import

shuffle = shuffle_op._shuffle_op.shuffle
