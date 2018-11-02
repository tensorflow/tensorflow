# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for error_handlers module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast

from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import error_handlers
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.core import errors
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ErrorHandlersTest(converter_testing.TestCase):

  def test_basic(self):

    def test_fn():
      raise ValueError()

    with self.converted(test_fn, error_handlers, {}) as result:
      with self.assertRaises(errors.GraphConstructionError):
        # Here we just assert that the handler works.
        result.test_fn()

  def test_no_origin_annotation(self):

    def test_fn(x):
      a = 0
      if x:
        a = random_ops.random_normal((2, 3), mean=0.0, dtype=dtypes.int32)
      else:
        a = 0
      return a

    node, ctx = self.prepare(test_fn, {
        'random_ops': random_ops,
        'dtypes': dtypes
    })
    # To simulate a function without origin info we use the control flow
    # converter which adds a function that lacks origin info so we will not have
    # a wrapping try/except that reraises the NotImplementedError as a
    # GraphConstructionError.
    node = control_flow.transform(node, ctx)
    node = error_handlers.transform(node, ctx)
    # TODO(b/111562364): remove run_cond from traceback.
    test_fn_try_body = node.body[0].body
    true_fn_body = test_fn_try_body[1].body
    false_fn_body = test_fn_try_body[2].body
    self.assertNotIn(gast.Try, true_fn_body)
    self.assertNotIn(gast.Try, false_fn_body)


if __name__ == '__main__':
  test.main()
