# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ShardedVariable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util


def _load_and_run(
    model_dir,
    inputs,
    signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY):
  """Load a SavedModel into a TF 1.x-style graph and run `signature_key`."""
  graph = ops.Graph()
  with graph.as_default(), session_lib.Session() as session:
    meta_graph_def = loader.load(session, [tag_constants.SERVING], model_dir)
    signature = meta_graph_def.signature_def[signature_key]
    feed_dict = {}
    for arg_name in inputs.keys():
      input_tensor = session.graph.get_tensor_by_name(
          signature.inputs[arg_name].name)
      feed_dict[input_tensor] = inputs[arg_name]
    output_dict = {}
    for output_name, output_tensor_info in signature.outputs.items():
      output_dict[output_name] = session.graph.get_tensor_by_name(
          output_tensor_info.name)
    return session.run(output_dict, feed_dict=feed_dict)


class ShardedVariableTest(test.TestCase):

  def test_sharded_variable_simple(self):
    v0 = variables_lib.Variable([0])
    v1 = variables_lib.Variable([1])
    s = sharded_variable.ShardedVariable([v0, v1], name='s')
    self.assertEqual(s.variables[0], v0)
    self.assertEqual(s.variables[1], v1)
    self.assertEqual(s.shape.as_list(), [2])
    self.assertEqual(s.dtype, v0.dtype)
    self.assertEqual(s.name, 's')

  def test_save_restore(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])
    cp.write(fname)

    self.evaluate(cp.s.variables[0].assign([4]))
    self.assertEqual(self.evaluate(cp.s.variables[0]), [4])

    cp.restore(fname)
    # Tests that the original weights are restored.
    self.assertEqual(self.evaluate(cp.s.variables[0]), [0])

  def test_save_restore_different_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')

    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [variables_lib.Variable([0, 0, 0, 0])]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')

    # Restore from 4 partitions into 1.
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1, 2, 3])

    self.evaluate(cp2.s.variables[0].assign([5, 10, 15, 20]))
    cp2.write(fname)

    # Restore 1 partition into 4.
    cp.restore(fname)
    self.assertEqual(self.evaluate(cp.s.variables[0]), [5])
    self.assertEqual(self.evaluate(cp.s.variables[1]), [10])
    self.assertEqual(self.evaluate(cp.s.variables[2]), [15])
    self.assertEqual(self.evaluate(cp.s.variables[3]), [20])

  def test_save_restore_4_to_2_partitions(self):
    fname = os.path.join(self.get_temp_dir(), 'checkpoint')
    variables = [
        variables_lib.Variable([0]),
        variables_lib.Variable([1]),
        variables_lib.Variable([2]),
        variables_lib.Variable([3])
    ]
    s = sharded_variable.ShardedVariable(variables, name='s')
    cp = util.Checkpoint(s=s)
    cp.write(fname)

    variables2 = [
        variables_lib.Variable([0, 0]),
        variables_lib.Variable([0, 0])
    ]
    s2 = sharded_variable.ShardedVariable(variables2, name='s')
    cp2 = util.Checkpoint(s=s2)
    cp2.restore(fname)
    # Assert that weights from the 4 partitions were loaded here.
    self.assertLen(cp2.s.variables, 2)
    self.assertAllEqual(self.evaluate(cp2.s.variables[0]), [0, 1])
    self.assertAllEqual(self.evaluate(cp2.s.variables[1]), [2, 3])

  def test_save_graph_def(self):
    root = tracking.AutoTrackable()
    v1 = variables_lib.Variable([3.])
    v2 = variables_lib.Variable([2.])
    root.v = sharded_variable.ShardedVariable([v1, v2])
    root.train = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables, x))
    # TODO(b/144057383): Remove the necessity of root.serve once saving context
    # is made to tf.function cache.
    root.serve = def_function.function(
        lambda x: embedding_ops.embedding_lookup_v2(root.v.variables[0], x),
        input_signature=[tensor_spec.TensorSpec([2], dtypes.int32, name='x')])

    # Trace and use root.train
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

    save_dir = os.path.join(self.get_temp_dir(), 'saved_model')
    save.save(root, save_dir, root.serve)
    self.assertAllEqual([3., 2.],
                        _load_and_run(save_dir, {'x': [0, 1]})['output_0'])

    # Continue using root.train for training
    self.assertAllEqual([3., 2.], root.train([0, 1]).numpy())

  def test_validation_errors(self):
    with self.assertRaisesRegex(ValueError, 'Expected a list of '):
      sharded_variable.ShardedVariable(
          [variables_lib.Variable([0]), 'not-a-variable'])

    with self.assertRaisesRegex(ValueError, 'must have the same dtype'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable([0], dtype='int64'),
          variables_lib.Variable([1], dtype='int32')
      ])

    with self.assertRaisesRegex(ValueError, 'the same shapes except'):
      sharded_variable.ShardedVariable([
          variables_lib.Variable(array_ops.ones((5, 10))),
          variables_lib.Variable(array_ops.ones((5, 20)))
      ])

    with self.assertRaisesRegex(ValueError, '`SaveSliceInfo` should not'):
      v = variables_lib.Variable([0])
      v._set_save_slice_info(
          variables_lib.Variable.SaveSliceInfo(
              full_name='s', full_shape=[2], var_offset=[0], var_shape=[1]))
      sharded_variable.ShardedVariable([v])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
