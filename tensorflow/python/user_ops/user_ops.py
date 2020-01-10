"""All user ops."""

import tensorflow.python.platform
from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *


def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()
