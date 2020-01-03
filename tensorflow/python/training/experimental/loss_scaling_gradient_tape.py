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
"""Contains Loss Scale Gradient Tape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import backprop
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import nest


def _convert_to_per_replica(distribution, value):
  """Converts a tensor or a DistributedVariable to a PerReplica value."""
  return distribution.experimental_run_v2(array_ops.identity, args=(value,))


# TODO(reedwm): Expose this after testing it on several models.
class LossScaleGradientTape(backprop.GradientTape):
  """A gradient tape that scales losses and unscales resulting gradients.

  Operates as a normal gradient tape, but takes in a
  `tf.mixed_precision.experimental.LossScale` object. Losses are scaled up by
  some amount before the gradients are calculated and the resulting gradients
  are scaled down by the same amount.

  This has no net mathematical effect, but can be used to prevent vanishing
  gradients, for example in the case of mixed precision training.

  If a DynamicLossScale object is used and non-finite gradients are encountered,
  the loss scale will be updated and the gradients recomputed until either
  finite gradients are encountered or the loss scale becomes 1.

  This class should *not* be used with a LossScaleOptimizer, as both classes
  update the LossScale object. Use a non-loss scaling optimizer instead.

  Usage:
  ```
  opt = tf.keras.optimizers.SGD(1.0)
  model_loss_scale = tf.mixed_precision.experimental.DynamicLossScale()

  for step in training_steps:
    with LossScaleGradientTape(model_loss_scale) as tape:
      logits = ...  # Run model and get logits
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=labels)
      loss = tf.reduce_mean(loss)
    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    opt.apply_gradients(zip(grads, vars))
  ```

  WARNING: Computing second-order (or higher) gradients with a
  `LossScaleGradientTape` does not yet work properly when a
  `tf.distribute.Strategy` is used. Computing second-order gradients will return
  None instead of the gradient tensors. This only occurs when you nest multiple
  gradient tapes under each other; if you do not nest them, this issue will not
  occur.
  """

  def __init__(self,
               loss_scale,
               persistent=False,
               watch_accessed_variables=True):
    """Creates a new LossScaleGradientTape.

    Args:
      loss_scale: `tf.mixed_precision.experimental.LossScale` object that
        manages what quantity to scale by. This is typically either a
        FixedLossScale object with a constant scalar or a
        `tf.mixed_precision.experimental.DynamicLossScale` object that will
        adjust the scalar appropriately if any non-finite gradients are
        encountered.
      persistent: Boolean controlling whether a persistent gradient tape is
        created. False by default, which means at most one call can be made to
        the gradient() method on this object.
      watch_accessed_variables: Boolean controlling whether the tape will
        automatically `watch` any (trainable) variables accessed while the tape
        is active. Defaults to True meaning gradients can be requested from any
        result computed in the tape derived from reading a trainable `Variable`.
        If False users must explicitly `watch` any `Variable`s they want to
        request gradients from.
    """
    if not isinstance(loss_scale, loss_scale_module.LossScale):
      raise ValueError("`loss_scale` must be an instance of LossScale, "
                       "but got: %s" % (loss_scale,))
    if not ops.executing_eagerly_outside_functions():
      raise ValueError("LossScaleGradientTape is only supported in Eager mode.")

    # always make a persistent tape to loop over loss scaling
    super(LossScaleGradientTape, self).__init__(True,
                                                watch_accessed_variables)
    self._outer_persistent = persistent
    self._loss_scale = loss_scale

  def gradient(self,
               target,
               sources,
               output_gradients=None,
               unconnected_gradients=UnconnectedGradients.NONE):
    """Computes the gradient using operations recorded in context of this tape.

    Uses the `LossScale` object provided in the constructor to scale `target`
    and then to unscale the resulting gradients.

    Args:
      target: a list or nested structure of Tensors or Variables to be
        differentiated.
      sources: a list or nested structure of Tensors or Variables. `target` will
        be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of target.
        Defaults to None.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`. If non-finite gradients are encountered
      after dynamic scaling, the loss scale will be updated and the gradients
      recomputed until either finite gradients are encountered or the loss scale
      becomes 1.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
      ValueError: if the target is a variable or if unconnected gradients is
       called with an unknown value.
    """
    if self._tape is None:  # pylint: disable=access-member-before-definition
      raise RuntimeError("GradientTape.gradient can only be called once on "
                         "non-persistent tapes.")
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError("LossScaleGradientTape.gradient() must be called in a "
                       "replica context.")

    # Note: DistributionStrategy does not support running a while loop in a
    # replica context. So, we call `_compute_gradients_until_finite` in a cross-
    # replica context.
    replica_context = distribution_strategy_context.get_replica_context()
    grads = replica_context.merge_call(
        _compute_gradients_until_finite,
        args=(self, self._loss_scale, target, sources, output_gradients,
              unconnected_gradients))

    if not self._outer_persistent:
      self._tape = None  # free up resources if a persistent tape was not needed
    return grads

  def jacobian(self,
               target,
               sources,
               unconnected_gradients=UnconnectedGradients.NONE,
               parallel_iterations=None,
               experimental_use_pfor=True):
    # TODO(reedwm): Implement this
    raise NotImplementedError("LossScaleGradientTape.jacobian is not "
                              "yet implemented")

  def batch_jacobian(self,
                     target,
                     source,
                     unconnected_gradients=UnconnectedGradients.NONE,
                     parallel_iterations=None,
                     experimental_use_pfor=True):
    # TODO(reedwm): Implement this
    raise NotImplementedError("LossScaleGradientTape.batch_jacobian is not "
                              "yet implemented")


def _compute_gradients_until_finite(
    distribution, loss_scale_gradient_tapes, loss_scale, target, sources,
    output_gradients, unconnected_gradients):
  """Compute gradients and update the loss scale until the gradients are finite.

  This must be called in a cross-replica context.

  This is a function instead of a method of LossScaleGradientTape, as the `self`
  parameter would be meaningless. There is one LossScaleGradientTape per
  replica, but this function is called once total (not per replica), so there
  cannot be a singular `self` parameter.

  Args:
    distribution: The distribution strategy in effect.
    loss_scale_gradient_tapes: A PerReplica value of LossScaleGradientTapes.
      Contains the LossScaleGradientTape of each replica.
    loss_scale: The loss scale to use to scale the loss and unscale the
      gradient.
    target: a list or nested structure of Tensors or Variables to be
      differentiated.
    sources: a list or nested structure of Tensors or Variables. `target` will
      be differentiated against elements in `sources`.
    output_gradients: Passed to GradientTape.gradient
    unconnected_gradients: Pass to GradientTape.gradient.

  Returns:
    The gradients of `target` with respect to `sources`.
  """
  # Autograph cannot convert this function, so we must use an explicit
  # tf.while_loop.
  # TODO(b/143572314): Fix Autograph so that it can convert this function, then
  # replace the tf.while_loop with a Python while loop.

  # For convenience, we only deal with flattened sources
  flattened_sources = nest.flatten(sources)

  # Define the initial loop variables of the while loop.

  # Dummy value for initial_grads. The first iteration of the loop will
  # overwrite `grads` to the actual gradients.
  initial_grads = flattened_sources
  if distribution_strategy_context.has_strategy():
    # A while_loop requires the initial values to have the same types as the
    # return values from the body. However, 'initial_grads' may have type
    # 'DistributionVariable', while body returns a 'PerReplica'. While both
    # types subclass 'DistributedValues', while_loop will still throw an error.
    # So we convert 'initial_grads' to be PerReplica values.
    # TODO(b/146084534): Once the bug is fixed, remove this special case.
    initial_grads = [_convert_to_per_replica(distribution, g)
                     for g in initial_grads]
  initial_ready_to_update = False
  initial_is_first_iteration = True

  def cond(grads, ready_to_update, is_first_iteration):
    """The condition of the while loop."""
    del grads
    # Equivalent to:
    # `is_first_iteration or (not ready_to_update and loss_scale() > 1)`
    return math_ops.logical_or(
        is_first_iteration,
        math_ops.logical_and(
            math_ops.logical_not(ready_to_update),
            math_ops.greater(loss_scale(), 1)))

  # Boolean list specifying whether each gradient is None or not. Set by body().
  is_nones = []

  def body(grads, ready_to_update, is_first_iteration):
    """The body of the while loop."""
    del grads, ready_to_update, is_first_iteration
    def replica_fn(gradient_tape, target, flattened_sources, output_gradients):
      """Scales the loss, computes the gradients, and unscales the gradients."""
      loss_scale_val = loss_scale()
      with gradient_tape:  # re-enter gradient tape so it sees the loss scaling
        scaled_target = nest.map_structure(
            lambda t: t * math_ops.cast(loss_scale_val, t.dtype), target)
      scaled_grads = super(LossScaleGradientTape, gradient_tape).gradient(
          scaled_target, flattened_sources, output_gradients,
          unconnected_gradients)

      is_nones[:] = [g is None for g in scaled_grads]
      inv_loss_scale = 1.0 / loss_scale_val
      grads = []  # The unscaled gradients
      for g, initial_grad in zip(scaled_grads, initial_grads):
        if g is not None:
          grads.append(g * math_ops.cast(inv_loss_scale, g.dtype))
        else:
          # We cannot return None from a tf.while_loop, so we pass a dummy
          # tensor instead. We use initial_grad as a dummy tensor as it has the
          # correct shape and dtype. We replace it with None outside the while
          # loop.
          grads.append(initial_grad)
      return grads

    # Switch to a replica-context to compute gradients once per replica.
    grads = distribution.experimental_run_v2(
        replica_fn, args=(loss_scale_gradient_tapes, target, flattened_sources,
                          output_gradients))
    # Check for non-finite gradients possibly resulting from scaling.
    _, ready_to_update = loss_scale.update(grads)
    is_first_iteration = False
    return grads, ready_to_update, is_first_iteration

  grads, _, _ = control_flow_ops.while_loop(
      cond, body, [initial_grads, initial_ready_to_update,
                   initial_is_first_iteration])
  grads = [None if is_none else g for g, is_none in zip(grads, is_nones)]
  grads = nest.pack_sequence_as(sources, grads)
  return grads
