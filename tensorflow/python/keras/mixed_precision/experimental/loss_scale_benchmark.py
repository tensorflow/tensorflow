# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for LossScaleOptimizer and LossScaleGradientTape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scaling_gradient_tape as lsgt_module


def _get_strategy(num_gpus):
  if num_gpus > 1:
    return mirrored_strategy.MirroredStrategy(
        ['/GPU:%d' % i for i in range(num_gpus)])
  else:
    return distribution_strategy_context.get_strategy()  # The default strategy


class LossScaleBenchmark(test.Benchmark):
  """Benchmark for loss scaling."""

  def _benchmark(self, gradient_type, num_gpus, mode, loss_scaling):
    """Benchmarks loss scaling.

    We run a simple model with several scalar variables. The loss is the sum of
    all variables. The model is simple because we want to measure only the
    performance of loss scaling, not the performance of the model itself.

    Args:
      gradient_type: "optimizer" or "gradient_tape". How gradients are computed.
        "optimizer" uses Optimizer.minimize. "gradient_tape" uses
        GradientTape.gradient.
      num_gpus: The number of GPUs to use. Must be at least 1.
      mode: "eager", "tf_function", or "graph". "eager" means to use eager mode.
        "tf_function" means to use eager mode where all computations are wrapped
        in a tf.function. "graph" means to use TensorFlow 1's graph mode with a
        tf.compat.v1.Session. "graph" is unsupported with a
        LossScaleGradientTape.
      loss_scaling: "fixed", "dynamic", or None. The type of loss scaling to
        use. None means use no loss scaling, which is useful as a baseline to
        see how much slower loss scaling is in comparison.
    """
    if mode == 'graph':
      graph = ops.Graph()
      ctx_mgr = graph.as_default()
    elif mode == 'eager':
      ctx_mgr = context.eager_mode()
    else:
      assert mode == 'tf_function'
      ctx_mgr = context.eager_mode()
    ls_str = loss_scaling or 'no_loss_scaling'
    name = '%s_%d_GPU_%s_%s' % (gradient_type, num_gpus, mode, ls_str)
    with ctx_mgr, _get_strategy(num_gpus).scope() as strategy:
      opt = adam.Adam()
      if loss_scaling == 'fixed':
        loss_scale = loss_scale_module.FixedLossScale(2.)
      elif loss_scaling == 'dynamic':
        # Make increment_period so high that it's effectively infinite. This
        # means the loss scale will never change. Any performance overhead
        # from increasing/decreasing the loss scale is typically negligible
        # since it happens infrequently, so we only benchmark the common case
        # of the loss scale not changing.
        increment_period = 1000000
        loss_scale = loss_scale_module.DynamicLossScale(
            initial_loss_scale=2., increment_period=increment_period)
      else:
        assert loss_scaling is None
        loss_scale = None

      num_vars = 200
      num_warmup_iters = 1
      num_iters = 20
      # By using scalar variables, we reduce overhead of the actual GPU work of
      # multiplying variables, dividing gradients, and checking gradients for
      # NaNs. Measuring these overheads isn't very useful as there is little we
      # can do to reduce them (one such way would be to fuse dividing gradients
      # and checking them for NaNs). We still have all other overheads, such as
      # all-reducing the `is_finite` values and having a tf.cond or
      # tf.while_loop based on whether gradients are NaNs. Currently, these
      # other overheads are much more significant than the GPU work.
      var_list = [
          variables.Variable(i, dtype='float32') for i in range(num_vars)]

      def get_loss():
        return math_ops.add_n(var_list)

      if gradient_type == 'gradient_tape':
        tape_cls = ((lambda: lsgt_module.LossScaleGradientTape(loss_scale))
                    if loss_scale else backprop.GradientTape)
        def minimize_fn():
          with tape_cls() as tape:
            loss = get_loss()
          grads = tape.gradient(loss, var_list)
          return opt.apply_gradients(zip(grads, var_list))
      else:
        assert gradient_type == 'optimizer'
        if loss_scale:
          opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
        def minimize_fn():
          return opt.minimize(get_loss, var_list)

      if mode == 'graph':
        run_op = strategy.experimental_run_v2(minimize_fn)
        init_op = variables.global_variables_initializer()
        with session_module.Session() as sess:
          sess.run(init_op)
          self.run_op_benchmark(sess, run_op, min_iters=num_iters,
                                burn_iters=num_warmup_iters, name=name)
        return

      def run_fn():
        strategy.experimental_run_v2(minimize_fn)
      if mode == 'tf_function':
        run_fn = def_function.function(run_fn)

      for _ in range(num_warmup_iters):
        run_fn()

      start = time.time()
      for _ in range(num_iters):
        run_fn()
      end = time.time()
      self.report_benchmark(iters=num_iters,
                            wall_time=(end - start) / num_iters, name=name)

  def _gpus_to_test_with(self):
    num_gpus = context.num_gpus()
    gpus_to_test_with = []
    if num_gpus >= 1:
      gpus_to_test_with.append(1)
    if num_gpus >= 2:
      gpus_to_test_with.append(2)
    if num_gpus >= 8:
      gpus_to_test_with.append(8)
    return gpus_to_test_with

  def benchmark_optimizer(self):
    for num_gpus in self._gpus_to_test_with():
      for mode in 'eager', 'tf_function', 'graph':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('optimizer', num_gpus, mode, loss_scaling)

  def benchmark_gradient_tape(self):
    for num_gpus in self._gpus_to_test_with():
      # LossScaleGradientTape doesn't support graph mode
      for mode in 'eager', 'tf_function':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('gradient_tape', num_gpus, mode, loss_scaling)


if __name__ == '__main__':
  test.main()
