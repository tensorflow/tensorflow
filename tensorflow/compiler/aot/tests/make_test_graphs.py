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
"""Generate tensorflow graphs for testing tfcompile."""

import argparse
import os
import sys

from absl import app
import six
from six.moves import range

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import saver as saver_lib

FLAGS = None


def tfadd(_):
  x = constant_op.constant([1], name='x_const')
  y = constant_op.constant([2], name='y_const')
  math_ops.add(x, y, name='x_y_sum')


def tfadd_with_ckpt(out_dir):
  x = array_ops.placeholder(dtypes.int32, name='x_hold')
  y = variables.VariableV1(constant_op.constant([0]), name='y_saved')
  math_ops.add(x, y, name='x_y_sum')

  init_op = variables.global_variables_initializer()
  saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V1)
  with session.Session() as sess:
    sess.run(init_op)
    sess.run(y.assign(y + 42))
    # Without the checkpoint, the variable won't be set to 42.
    ckpt = os.path.join(out_dir, 'test_graph_tfadd_with_ckpt.ckpt')
    saver.save(sess, ckpt)


def tfadd_with_ckpt_saver(out_dir):
  x = array_ops.placeholder(dtypes.int32, name='x_hold')
  y = variables.VariableV1(constant_op.constant([0]), name='y_saved')
  math_ops.add(x, y, name='x_y_sum')

  init_op = variables.global_variables_initializer()
  saver = saver_lib.Saver(name='abcprefix', write_version=saver_pb2.SaverDef.V1)
  with session.Session() as sess:
    sess.run(init_op)
    sess.run(y.assign(y + 42))
    # Without the checkpoint, the variable won't be set to 42.
    ckpt_file = os.path.join(out_dir, 'test_graph_tfadd_with_ckpt_saver.ckpt')
    saver.save(sess, ckpt_file)
    # Without the SaverDef, the restore op won't be named correctly.
    saver_file = os.path.join(out_dir, 'test_graph_tfadd_with_ckpt_saver.saver')
    with open(saver_file, 'wb') as f:
      f.write(six.ensure_binary(saver.as_saver_def().SerializeToString()))


def tfassert_eq(_):
  x = array_ops.placeholder(dtypes.int32, name='x_hold')
  y = array_ops.placeholder(dtypes.int32, name='y_hold')
  control_flow_assert.Assert(
      math_ops.equal(x, y), ['Expected x == y.'], name='assert_eq')
  math_ops.add(x, math_ops.negative(y), name='x_y_diff')


def tfcond(_):
  p = array_ops.placeholder(dtypes.bool, name='p_hold')
  x = array_ops.placeholder(dtypes.int32, name='x_hold')
  y = array_ops.placeholder(dtypes.int32, name='y_hold')
  z = control_flow_ops.cond(p, lambda: x, lambda: y)
  array_ops.identity(z, name='result')


def tfgather(_):
  params = array_ops.placeholder(dtypes.float32, name='params')
  indices = array_ops.placeholder(dtypes.int32, name='indices')
  array_ops.gather(params, indices, name='gather_output')


def tfmatmul(_):
  x = array_ops.placeholder(dtypes.float32, name='x_hold')
  y = array_ops.placeholder(dtypes.float32, name='y_hold')
  math_ops.matmul(x, y, name='x_y_prod')


def tfmatmulandadd(_):
  # This tests multiple outputs.
  x = array_ops.placeholder(dtypes.float32, name='x_hold')
  y = array_ops.placeholder(dtypes.float32, name='y_hold')
  math_ops.matmul(x, y, name='x_y_prod')
  math_ops.add(x, y, name='x_y_sum')


def tffunction(_):

  @function.Defun(dtypes.int32, dtypes.int32)
  def test_func(a, b):
    return a + b

  x = constant_op.constant([1], name='x_const')
  y = constant_op.constant([2], name='y_const')
  test_func(x, y, name='func_call')  # pylint: disable=unexpected-keyword-arg


def tfsplits(_):
  """A more complex graph, including splits."""
  x = array_ops.placeholder(dtypes.float32, shape=[2, 2], name='x')
  y = array_ops.placeholder(dtypes.float32, shape=[2, 2], name='y')
  for _ in range(3):
    x0, x1 = array_ops.split(x, 2, 0)
    y0, y1 = array_ops.split(y, 2, 0)
    x0 += 1
    y0 += 1
    z = math_ops.matmul(x, y, name='x_y_prod')
    a = array_ops.concat([x0, y1], axis=0, name='concat_x0_y1')
    b = array_ops.concat([y0, x1], axis=0, name='concat_y0_x1')
    x = math_ops.matmul(a, b, name='a_b')
    y = math_ops.add(x, z)
  array_ops.identity(y, name='result')


def tftop_k(_):
  x = array_ops.placeholder(dtypes.int32, shape=[5], name='x')
  output = nn_ops.top_k(x, 2, name='values')
  array_ops.identity(output[1], name='indices')


def tfvariable_readonly(_):
  x = variables.Variable(1000.0, name='x')
  unused_y = variables.Variable(1000.0, name='y')
  old_x = x.value()
  with ops.control_dependencies([old_x]):
    new_value = math_ops.add(old_x, 42.0)
  array_ops.identity(new_value, name='result')


# TODO(b/147908587): Change x and the two constants back to have a scalar shape
#                    when the bug is fixed.
def tfvariable(_):
  x = variables.Variable([1000.0], name='x', shape=[1])
  old_x = x.value()
  with ops.control_dependencies([old_x]):
    new_x = x.assign_add([42.0])
  array_ops_stack.stack([old_x, new_x], name='result')


def tfvariable_sequential_updates(_):
  x = variables.Variable(1.0, name='x')
  y = variables.Variable(1.0, name='y')
  updates = control_flow_ops.no_op()
  for _ in range(3):
    with ops.control_dependencies([updates]):
      x_val = x.read_value() + y
      updates = x.assign_sub(0.1 * x_val)

  array_ops.identity(updates, name='result')


def export_debug_info(exported_graph):
  """Exports debug information from a graph.

  Args:
    exported_graph: A Graph that has been created by tracing a saveable view.

  Returns:
    Corresponding GraphDebugInfo with traces for all ops in exported_graph.
  """
  exported_operations = []
  for op in exported_graph.get_operations():
    exported_operations.append(('', op))
  return error_interpolation.create_graph_debug_info_def(exported_operations)


def write_graph(build_graph, out_dir, debug_info=False):
  """Build a graph using build_graph and write it out."""
  g = ops.Graph()
  with g.as_default():
    build_graph(out_dir)
    filename = os.path.join(out_dir, 'test_graph_%s.pb' % build_graph.__name__)
    with open(filename, 'wb') as f:
      f.write(six.ensure_binary(g.as_graph_def().SerializeToString()))

    if debug_info:
      filename_debuginfo = os.path.join(
          out_dir, 'test_debuginfo_%s.pb' % build_graph.__name__)
      test_debuginfo = export_debug_info(g)
      with open(filename_debuginfo, 'wb') as f:
        f.write(
            six.ensure_binary(
                test_debuginfo.SerializeToString(deterministic=True)))


def main(_):
  control_flow_util.enable_control_flow_v2()
  write_graph(tfadd, FLAGS.out_dir, debug_info=True)
  write_graph(tfadd_with_ckpt, FLAGS.out_dir)
  write_graph(tfadd_with_ckpt_saver, FLAGS.out_dir)
  write_graph(tfassert_eq, FLAGS.out_dir)
  write_graph(tfcond, FLAGS.out_dir)
  write_graph(tffunction, FLAGS.out_dir)
  write_graph(tfgather, FLAGS.out_dir)
  write_graph(tfmatmul, FLAGS.out_dir)
  write_graph(tfmatmulandadd, FLAGS.out_dir)
  write_graph(tfsplits, FLAGS.out_dir)
  write_graph(tftop_k, FLAGS.out_dir)
  write_graph(tfvariable, FLAGS.out_dir)
  write_graph(tfvariable_readonly, FLAGS.out_dir)
  write_graph(tfvariable_sequential_updates, FLAGS.out_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--out_dir',
      type=str,
      default='',
      help='Output directory for graphs, checkpoints and savers.')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
