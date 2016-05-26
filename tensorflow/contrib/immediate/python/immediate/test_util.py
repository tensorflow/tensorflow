# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=invalid-name
"""Test utils for tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import math
import re
import sys
import threading

import numpy as np
import six

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util.protobuf import compare

from tensorflow.python.framework import test_util as tf_test_util

import tensorflow.contrib.immediate as immediate

class ImmediateTestCase(tf_test_util.TensorFlowTestCase):
  """Base class for tests that need to test TensorFlow in immediate mode.
  """

  def __init__(self, methodName="runTest"):
    super(ImmediateTestCase, self).__init__(methodName)
    self._cached_env = None

  def setUp(self):
    self._ClearCachedEnv()
    ops.reset_default_graph()

  def tearDown(self):
    self._ClearCachedEnv()

  def _ClearCachedEnv(self):
    if self._cached_env is not None:
      self._cached_env.close()
      self._cached_env = None

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def test_env(self, tf=None):
    """Returns a immediate Environment for use in executing tests.

    Example:

      class MyOperatorTest(test_util.TensorFlowTestCase):
        def testMyOperator(self):
          with self.test_env(tf):
            valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = MyOperator(valid_input)
            self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
            invalid_input = [-1.0, 2.0, 7.0]
            with self.assertRaisesOpError("negative input not supported"):
              MyOperator(invalid_input)

    Returns:
      An Env object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """

    def prepare_config():
      config = config_pb2.ConfigProto()
      config.graph_options.optimizer_options.opt_level = -1
      return config

    env = immediate.Env._get_global_default_env()
    if env is None:
      env = immediate.Env(tf, config=prepare_config())

    with env.g.as_default():
      yield env


# TODO(yaroslavvb): delete because not used?
class TensorFlowTestCase(ImmediateTestCase):
   """Class for porting tests that expect TensorFlowTestCase."""

   @contextlib.contextmanager
   def test_session(self, use_gpu=False):
     # restore graph from default env because default graph is used
     # by session ops to create handle deleters

     with immediate.Env._get_global_default_env().g.as_default():
       yield 1


