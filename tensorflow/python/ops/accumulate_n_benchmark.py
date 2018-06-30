# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmark for accumulate_n() in math_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test


class AccumulateNBenchmark(test.Benchmark):

  def _AccumulateNTemplate(self, inputs, init, shape, validate_shape):
    var = gen_state_ops.temporary_variable(
        shape=shape, dtype=inputs[0].dtype.base_dtype)
    ref = state_ops.assign(var, init, validate_shape=validate_shape)
    update_ops = [
        state_ops.assign_add(
            ref, tensor, use_locking=True).op for tensor in inputs
    ]
    with ops.control_dependencies(update_ops):
      return gen_state_ops.destroy_temporary_variable(ref, var_name=var.op.name)

  def _AccumulateNInitializedWithFirst(self, inputs):
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros_like(inputs[0]),
        shape=inputs[0].get_shape(),
        validate_shape=True)

  def _AccumulateNInitializedWithMerge(self, inputs):
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros_like(gen_control_flow_ops.merge(inputs)[0]),
        shape=tensor_shape.vector(0),
        validate_shape=False)

  def _AccumulateNInitializedWithShape(self, inputs):
    return self._AccumulateNTemplate(
        inputs,
        init=array_ops.zeros(
            shape=inputs[0].get_shape(), dtype=inputs[0].dtype.base_dtype),
        shape=inputs[0].get_shape(),
        validate_shape=True)

  def _GenerateUnorderedInputs(self, size, n):
    inputs = [random_ops.random_uniform(shape=[size]) for _ in xrange(n)]
    random.shuffle(inputs)
    return inputs

  def _GenerateReplicatedInputs(self, size, n):
    return n * self._GenerateUnorderedInputs(size, 1)

  def _GenerateOrderedInputs(self, size, n):
    inputs = self._GenerateUnorderedInputs(size, 1)
    queue = data_flow_ops.FIFOQueue(
        capacity=1, dtypes=[inputs[0].dtype], shapes=[inputs[0].get_shape()])
    for _ in xrange(n - 1):
      op = queue.enqueue(inputs[-1])
      with ops.control_dependencies([op]):
        inputs.append(math_ops.tanh(1.0 + queue.dequeue()))
    return inputs

  def _GenerateReversedInputs(self, size, n):
    inputs = self._GenerateOrderedInputs(size, n)
    inputs.reverse()
    return inputs

  def _SetupAndRunBenchmark(self, graph, inputs, repeats, format_args):
    with graph.as_default():
      add_n = math_ops.add_n(inputs)
      acc_n_first = self._AccumulateNInitializedWithFirst(inputs)
      acc_n_merge = self._AccumulateNInitializedWithMerge(inputs)
      acc_n_shape = self._AccumulateNInitializedWithShape(inputs)

    test_ops = (("AddN", add_n.op),
                ("AccNFirst", acc_n_first.op),
                ("AccNMerge", acc_n_merge.op),
                ("AccNShape", acc_n_shape.op))

    with session.Session(graph=graph):
      for tag, op in test_ops:
        for _ in xrange(100):
          op.run()  # Run for warm up.
        start = time.time()
        for _ in xrange(repeats):
          op.run()
        duration = time.time() - start
        args = format_args + (tag, duration)
        print(self._template.format(*args))

  def _RunBenchmark(self, tag, input_fn, sizes, ninputs, repeats):
    for size in sizes:
      for ninput in ninputs:
        graph = ops.Graph()
        with graph.as_default():
          inputs = input_fn(size, ninput)

        format_args = (tag, size, ninput, repeats)
        self._SetupAndRunBenchmark(graph, inputs, repeats, format_args)

  def benchmarkAccumulateN(self):
    self._template = "{:<15}" * 6
    args = {
        "sizes": (128, 128**2),
        "ninputs": (1, 10, 100, 300),
        "repeats": 100
    }
    benchmarks = (("Replicated", self._GenerateReplicatedInputs),
                  ("Unordered", self._GenerateUnorderedInputs),
                  ("Ordered", self._GenerateOrderedInputs),
                  ("Reversed", self._GenerateReversedInputs))

    print(self._template.format("", "Size", "#Inputs", "#Repeat", "Method",
                                "Duration"))
    print("-" * 90)
    for benchmark in benchmarks:
      self._RunBenchmark(*benchmark, **args)


if __name__ == "__main__":
  test.main()
