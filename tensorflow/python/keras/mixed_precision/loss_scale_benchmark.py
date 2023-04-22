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
"""Benchmarks for LossScaleOptimizer."""

import time

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module


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
        GradientTape.gradient along with LossScaleOptimizer.get_scaled_loss and
        LossScaleOptimizer.get_unscaled_gradients.
      num_gpus: The number of GPUs to use. Must be at least 1.
      mode: "eager" or "tf_function". "tf_function" causes all computations to
        be wrapped in a tf.function, while "eager" runs computations eagerly.
      loss_scaling: "fixed", "dynamic", or None. The type of loss scaling to
        use. None means use no loss scaling, which is useful as a baseline to
        see how much slower loss scaling is in comparison.
    """
    ls_str = loss_scaling or 'no_loss_scaling'
    name = '%s_%d_GPU_%s_%s' % (gradient_type, num_gpus, mode, ls_str)
    with context.eager_mode(), _get_strategy(num_gpus).scope() as strategy:
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
      if loss_scale:
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)

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
        if loss_scale is None:
          def minimize_fn():
            with backprop.GradientTape() as tape:
              loss = get_loss()
            grads = tape.gradient(loss, var_list)
            return opt.apply_gradients(zip(grads, var_list))
        else:
          def minimize_fn():
            with backprop.GradientTape() as tape:
              loss = get_loss()
              scaled_loss = opt.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, var_list)
            grads = opt.get_unscaled_gradients(scaled_grads)
            return opt.apply_gradients(zip(grads, var_list))
      else:
        assert gradient_type == 'optimizer'
        def minimize_fn():
          return opt.minimize(get_loss, var_list)

      def run_fn():
        strategy.run(minimize_fn)
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
    num_gpus = len(config.list_logical_devices('GPU'))
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
      for mode in 'eager', 'tf_function':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('optimizer', num_gpus, mode, loss_scaling)

  def benchmark_gradient_tape(self):
    for num_gpus in self._gpus_to_test_with():
      for mode in 'eager', 'tf_function':
        for loss_scaling in None, 'fixed', 'dynamic':
          self._benchmark('gradient_tape', num_gpus, mode, loss_scaling)


if __name__ == '__main__':
  test.main()
