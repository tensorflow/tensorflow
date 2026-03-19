# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Collective Operations with XLA."""

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test


class CollectiveOpXlaTest(test.TestCase):

  def testScopedAllocatorWithXla(self):
    group_size = 2
    group_key = 1
    instance_key1 = 1
    instance_key2 = 2
    tensor_size = 10

    graph_options = config_pb2.GraphOptions(
        optimizer_options=config_pb2.OptimizerOptions(
            do_constant_folding=False))
    cfg = config_pb2.ConfigProto(device_count={'CPU': group_size},
                                 graph_options=graph_options)
    rewrite_options = cfg.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    del rewrite_options.scoped_allocator_opts.enable_op[:]
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')

    # Tests that execute collectives need to be enclosed in graph or tf.function
    with ops.Graph().as_default(), self.session(config=cfg) as sess:
      run_ops = []
      for i in range(group_size):
        with ops.device('CPU:%d' % i):
          tensor_val = [i + 1.] * tensor_size
          constant = constant_op.constant(tensor_val)

          @def_function.function(jit_compile=True)
          def f(x):
            return 2 * x + 1

          input_tensor1 = array_ops.identity(f(constant))
          input_tensor2 = array_ops.identity(f(constant))
          reduced_tensor1 = collective_ops.all_reduce(
              input_tensor1, group_size, group_key, instance_key1, 'Add', 'Id')
          reduced_tensor2 = collective_ops.all_reduce(
              input_tensor2, group_size, group_key, instance_key2, 'Add', 'Id')
          run_ops.append(array_ops.identity(reduced_tensor1))
          run_ops.append(array_ops.identity(reduced_tensor2))
      results = sess.run(run_ops)
      for result in results:
        for result_val in result:
          self.assertEqual(result_val, 8.)


if __name__ == '__main__':
  test.main()
