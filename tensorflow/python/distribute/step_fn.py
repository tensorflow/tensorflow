# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""The step function abstraction represents a single training step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib


class Step(object):
  """Interface for performing each step of a training algorithm."""

  def __init__(self, distribution):
    self._distribution = distribution

  @property
  def distribution(self):
    return self._distribution

  def initialize(self):
    return []

  def __call__(self):
    """Perform one step of this training algorithm."""
    raise NotImplementedError("must be implemented in descendants")

  # TODO(priyag): Add an method to access initialization and finalize ops.


class StandardInputStep(Step):
  """Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  """

  def __init__(self, dataset_fn, distribution):
    super(StandardInputStep, self).__init__(distribution)
    self._iterator = distribution.make_input_fn_iterator(lambda _: dataset_fn())

  def initialize(self):
    return self._iterator.initializer


class StandardSingleLossStep(StandardInputStep):
  """A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(
      dataset, loss_fn, optimizer, distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that takes a context and inputs as arguments. It returns
      the loss for those inputs. `context` is an instance of
      `values.MultiStepContext` that will be passed when `loss_fn` is run.
      `context` can be used to specify the outputs to be returned from
      `loss_fn`, among other things.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  """

  def __init__(self, dataset_fn, loss_fn, optimizer, distribution,
               iterations_per_step=1):
    super(StandardSingleLossStep, self).__init__(dataset_fn, distribution)
    self._loss_fn = loss_fn
    self._optimizer = optimizer
    self._iterations_per_step = iterations_per_step

  def __call__(self):
    with self._distribution.scope():
      def step_fn(ctx, inputs):
        """Function to run one iteration with one input."""
        gradients_fn = backprop.implicit_grad(self._loss_fn)
        gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)

        grads_and_vars = self.distribution.extended.call_for_each_replica(
            gradients_fn, args=(ctx, inputs))
        # If threads use layers, then we need to run the first step
        # sequentially, so that layers.build() is not executed in parallel.
        # Otherwise, multiple sets of mirrored variables are going to be
        # created.
        return self._optimizer._distributed_apply(  # pylint: disable=protected-access
            self.distribution, grads_and_vars)

      # TODO(priyag): Return the outputs, context, etc as well.
      ctx = self.distribution.extended.experimental_run_steps_on_iterator(
          step_fn, self._iterator, self._iterations_per_step)
      return ctx.run_op
