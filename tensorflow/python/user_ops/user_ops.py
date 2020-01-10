"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *


def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()
