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

  def __call__(self):
    """Perform one step of this training algorithm."""
    return self.step(self.inputs())

  def inputs(self):
    """For the generating the input to be passed to `step()`."""
    raise NotImplementedError("must be implemented in descendants")

  def step(self, inputs):
    """Perform the main computation of this training algorithm."""
    raise NotImplementedError("must be implemented in descendants")


class StandardInputStep(Step):
  """Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  """

  def __init__(self, dataset_fn, distribution):
    Step.__init__(self, distribution)
    self._distributed_input = distribution.distribute_dataset(
        dataset_fn).make_one_shot_iterator()

  def inputs(self):
    return self._distributed_input.get_next()


class StandardSingleLossStep(StandardInputStep):
  """A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(dataset, loss_fn, optimizer)
  step.initialize(distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that returns loss.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  """

  def __init__(self, dataset_fn, loss_fn, optimizer, distribution):
    StandardInputStep.__init__(self, dataset_fn, distribution)
    self._loss_fn = loss_fn
    self._optimizer = optimizer
    self._is_run_concurrently = False

  def step(self, inputs):
    with self._distribution.scope():
      gradients_fn = backprop.implicit_grad(self._loss_fn)
      gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)

      grads_and_vars = self.distribution.call_for_each_tower(
          gradients_fn, inputs, run_concurrently=self._is_run_concurrently)
      # If threads use layers, then we need to run the first step sequentially,
      # so that layers.build() is not executed in parallel.  Otherwise, multiple
      # sets of mirrored variables are going to be created.
      self._is_run_concurrently = True
      return self._optimizer._distributed_apply(  # pylint: disable=protected-access
          self.distribution, grads_and_vars)
