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
"""Error traceback rewriting integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import autograph as ag
from tensorflow.python.util import tf_inspect


class ErrorsTest(tf.test.TestCase):

  def test_graph_construction_error_rewriting_call_tree(self):

    def innermost(x):
      if x > 0:
        return tf.random_normal((2, 3), mean=0.0, dtype=tf.int32)
      return tf.zeros((2, 3))

    def inner_caller():
      return innermost(1.0)

    def caller():
      return inner_caller()

    with self.assertRaises(ag.GraphConstructionError) as error:
      graph = ag.to_graph(caller)
      graph()
    expected = error.exception
    custom_traceback = expected.custom_traceback
    found_correct_filename = False
    num_innermost_names = 0
    num_inner_caller_names = 0
    num_caller_names = 0
    ag_output_filename = tf_inspect.getsourcefile(graph)
    for frame in custom_traceback:
      filename, _, fn_name, _ = frame
      self.assertFalse('control_flow_ops.py' in filename)
      self.assertFalse(ag_output_filename in filename)
      found_correct_filename |= __file__ in filename
      self.assertNotEqual('tf__test_fn', fn_name)
      num_innermost_names += int('innermost' == fn_name)
      self.assertNotEqual('tf__inner_caller', fn_name)
      num_inner_caller_names += int('inner_caller' == fn_name)
      self.assertNotEqual('tf__caller', fn_name)
      num_caller_names += int('caller' == fn_name)
    self.assertTrue(found_correct_filename)
    self.assertEqual(num_innermost_names, 1)
    self.assertEqual(num_inner_caller_names, 1)
    self.assertEqual(num_caller_names, 1)

  def test_graph_construction_error_rewriting_class(self):

    class TestClass(object):

      def test_fn(self):
        return tf.random_normal((2, 3), mean=0.0, dtype=tf.int32)

      def inner_caller(self):
        return self.test_fn()

      def caller(self):
        return self.inner_caller()

    # Note we expect a TypeError here because the traceback will not be
    # rewritten for classes.
    with self.assertRaises(TypeError):
      graph = ag.to_graph(TestClass)
      graph().caller()

  def test_runtime_error_rewriting(self):

    def g(x, s):
      while tf.reduce_sum(x) > s:
        x //= 0
      return x

    def test_fn(x):
      return g(x, 10)

    compiled_fn = ag.to_graph(test_fn)

    with self.assertRaises(ag.TfRuntimeError) as error:
      with self.test_session() as sess:
        x = compiled_fn(tf.constant([4, 8]))
        with ag.improved_errors(compiled_fn):
          sess.run(x)
    expected = error.exception
    custom_traceback = expected.custom_traceback
    found_correct_filename = False
    num_test_fn_frames = 0
    num_g_frames = 0
    ag_output_filename = tf_inspect.getsourcefile(compiled_fn)
    for frame in custom_traceback:
      filename, _, fn_name, source_code = frame
      self.assertFalse(ag_output_filename in filename)
      self.assertFalse('control_flow_ops.py' in filename)
      self.assertFalse('ag__.' in fn_name)
      self.assertFalse('tf__g' in fn_name)
      self.assertFalse('tf__test_fn' in fn_name)
      found_correct_filename |= __file__ in filename
      num_test_fn_frames += int('test_fn' == fn_name and
                                'return g(x, 10)' in source_code)
      # This makes sure that the code is correctly rewritten from "x_1 //= 0" to
      # "x //= 0".
      num_g_frames += int('g' == fn_name and 'x //= 0' in source_code)
    self.assertTrue(found_correct_filename)
    self.assertEqual(num_test_fn_frames, 1)
    self.assertEqual(num_g_frames, 1)

  def test_runtime_error_rewriting_nested(self):

    def test_fn(x):

      def g(y):
        return y**2 // 0

      s = 0
      for xi in x:
        s += g(xi)
      return s

    compiled_fn = ag.to_graph(test_fn)

    # TODO(b/111408261): Nested functions currently do not rewrite correctly,
    # when they do we should change this test to check for the same traceback
    # properties as the other tests.  This should throw a runtime error with a
    # frame with "g" as the function name but because we don't yet add
    # try/except blocks to inner functions the name is "tf__g".
    with self.assertRaises(ag.TfRuntimeError) as error:
      with self.test_session() as sess:
        x = compiled_fn(tf.constant([4, 8]))
        with ag.improved_errors(compiled_fn):
          sess.run(x)
    expected = error.exception
    custom_traceback = expected.custom_traceback
    num_tf_g_frames = 0
    for frame in custom_traceback:
      _, _, fn_name, _ = frame
      self.assertNotEqual('g', fn_name)
      num_tf_g_frames += int('tf__g' == fn_name)
    self.assertEqual(num_tf_g_frames, 1)


if __name__ == '__main__':
  tf.test.main()
