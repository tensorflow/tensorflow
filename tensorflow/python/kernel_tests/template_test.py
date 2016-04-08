# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for make_template."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

import tensorflow as tf

from tensorflow.python.ops import template


def var_scoped_function():
  return tf.get_variable("dummy",
                         shape=[1],
                         initializer=tf.zeros_initializer)


def internally_var_scoped_function(scope_name):
  with tf.variable_scope(scope_name):
    return tf.get_variable("dummy",
                           shape=[1],
                           initializer=tf.zeros_initializer)


def function_with_create(trainable):
  """Creates a variable as a side effect using tf.Variable."""
  tf.Variable(0, trainable=trainable)
  return tf.get_variable("dummy",
                         shape=[1],
                         initializer=tf.zeros_initializer)


class TemplateTest(tf.test.TestCase):

  def test_end_to_end(self):
    """This test shows a very simple line model with test_loss.

    The template is used to share parameters between a training and test model.
    """
    # y = 2x + 1
    training_input, training_output = ([1., 2., 3., 4.], [2.8, 5.1, 7.2, 8.7])
    test_input, test_output = ([5., 6., 7., 8.], [11, 13, 15, 17])

    tf.set_random_seed(1234)

    def test_line(x):
      m = tf.get_variable("w", shape=[],
                          initializer=tf.truncated_normal_initializer())
      b = tf.get_variable("b", shape=[],
                          initializer=tf.truncated_normal_initializer())
      return x * m + b

    line_template = template.make_template("line", test_line)

    train_prediction = line_template(training_input)
    test_prediction = line_template(test_input)

    train_loss = tf.reduce_mean(tf.square(train_prediction - training_output))
    test_loss = tf.reduce_mean(tf.square(test_prediction - test_output))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = optimizer.minimize(train_loss)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      initial_test_loss = sess.run(test_loss)
      sess.run(train_op)
      final_test_loss = sess.run(test_loss)

    # Parameters are tied, so the loss should have gone down when we trained it.
    self.assertLess(final_test_loss, initial_test_loss)

  def test_skip_stack_frames(self):
    first = traceback.format_stack()
    second = traceback.format_stack()
    result = template._skip_common_stack_elements(first, second)
    self.assertEqual(1, len(result))
    self.assertNotEqual(len(first), len(result))

  def test_template_with_name(self):
    tmpl1 = template.make_template("s1", var_scoped_function)
    tmpl2 = template.make_template("s1", var_scoped_function)

    v1 = tmpl1()
    v2 = tmpl1()
    v3 = tmpl2()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("s1/dummy:0", v1.name)
    self.assertEqual("s1_2/dummy:0", v3.name)

  def test_template_in_scope(self):
    tmpl1 = template.make_template("s1", var_scoped_function)
    tmpl2 = template.make_template("s1", var_scoped_function)

    with tf.variable_scope("scope"):
      v1 = tmpl1()
      v3 = tmpl2()

    # The template contract requires the following to ignore scope2.
    with tf.variable_scope("scope2"):
      v2 = tmpl1()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("scope/s1/dummy:0", v1.name)
    self.assertEqual("scope/s1_1/dummy:0", v3.name)

  def test_template_with_internal_reuse(self):
    tmpl1 = template.make_template("s1", internally_var_scoped_function)
    tmpl2 = template.make_template("s1", internally_var_scoped_function)

    v1 = tmpl1("test")
    v2 = tmpl1("test")
    v3 = tmpl2("test")
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("s1/test/dummy:0", v1.name)
    self.assertEqual("s1_2/test/dummy:0", v3.name)

    with self.assertRaises(ValueError):
      tmpl1("not_test")

  def test_template_without_name(self):
    with self.assertRaises(ValueError):
      template.make_template(None, var_scoped_function)

  def test_make_template(self):
    # Test both that we can call it with positional and keywords.
    tmpl1 = template.make_template(
        "s1", internally_var_scoped_function, scope_name="test")
    tmpl2 = template.make_template(
        "s1", internally_var_scoped_function, scope_name="test")

    v1 = tmpl1()
    v2 = tmpl1()
    v3 = tmpl2()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("s1/test/dummy:0", v1.name)
    self.assertEqual("s1_2/test/dummy:0", v3.name)

  def test_enforces_no_extra_trainable_variables(self):
    tmpl = template.make_template("s", function_with_create, trainable=True)

    tmpl()
    with self.assertRaises(ValueError):
      tmpl()

  def test_permits_extra_non_trainable_variables(self):
    tmpl = template.make_template("s", function_with_create, trainable=False)
    self.assertEqual(tmpl(), tmpl())

  def test_internal_variable_reuse(self):
    def nested():
      with tf.variable_scope("nested") as vs:
        v1 = tf.get_variable("x", initializer=tf.zeros_initializer, shape=[])
      with tf.variable_scope(vs, reuse=True):
        v2 = tf.get_variable("x")
      self.assertEqual(v1, v2)
      return v1

    tmpl1 = template.make_template("s1", nested)
    tmpl2 = template.make_template("s1", nested)

    v1 = tmpl1()
    v2 = tmpl1()
    v3 = tmpl2()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("s1/nested/x:0", v1.name)
    self.assertEqual("s1_2/nested/x:0", v3.name)

  def test_nested_templates(self):
    def nested_template():
      nested1 = template.make_template("nested", var_scoped_function)
      nested2 = template.make_template("nested", var_scoped_function)
      v1 = nested1()
      v2 = nested2()
      self.assertNotEqual(v1, v2)
      return v2

    tmpl1 = template.make_template("s1", nested_template)
    tmpl2 = template.make_template("s1", nested_template)

    v1 = tmpl1()
    v2 = tmpl1()
    v3 = tmpl2()
    self.assertEqual(v1, v2)
    self.assertNotEqual(v1, v3)
    self.assertEqual("s1/nested_1/dummy:0", v1.name)
    self.assertEqual("s1_2/nested_1/dummy:0", v3.name)


if __name__ == "__main__":
  tf.test.main()
