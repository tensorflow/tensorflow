# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Example of using multiplex op in a SavedModel.

multiplex_2_save.py and multiplex_4_load_use.py are programs that use this.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""

import tensorflow as tf


def _get_example_tensors():
  cond = tf.constant([True, False, True, False, True], dtype=bool)
  a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
  b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
  return cond, a, b


def save(multiplex_op, path):
  """Save a model that contains the given `multiplex_op`.

  Args:
    multiplex_op: A multiplex Custom Op, e.g. multiplex_4_op.multiplex. This is
      parameterized so it can also be used to create an "old" model with an
      older version of the op, e.g. multiplex_2_op.multiplex.
    path: Directory to save model to.
  """
  example_cond, example_a, example_b = _get_example_tensors()

  class UseMultiplex(tf.Module):

    @tf.function(input_signature=[
        tf.TensorSpec.from_tensor(example_cond),
        tf.TensorSpec.from_tensor(example_a),
        tf.TensorSpec.from_tensor(example_b)
    ])
    def use_multiplex(self, cond, a, b):
      return multiplex_op(cond, a, b)

  model = UseMultiplex()
  tf.saved_model.save(
      model,
      path,
      signatures=model.use_multiplex.get_concrete_function(
          tf.TensorSpec.from_tensor(example_cond),
          tf.TensorSpec.from_tensor(example_a),
          tf.TensorSpec.from_tensor(example_b)))


def load_and_use(path):
  """Load and used a model that was previously created by `save()`.

  Args:
    path: Directory to load model from, typically the same directory that was
      used by save().

  Returns:
    A tensor that is the result of using the multiplex op that is
    tf.constant([1, 20, 3, 40, 5], dtype=tf.int64).
  """
  example_cond, example_a, example_b = _get_example_tensors()
  restored = tf.saved_model.load(path)
  return restored.use_multiplex(example_cond, example_a, example_b)
