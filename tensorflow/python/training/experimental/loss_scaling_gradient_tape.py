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

from tensorflow.python.eager import backprop
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


@tf_export("mixed_precision.experimental.LossScaleGradientTape", v1=[])
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
      raise ValueError("`loss_scale` must be an instance of LossScale.")

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

    ready_to_update = False
    grads = nest.map_structure(array_ops.zeros_like, sources)

    while not ready_to_update and self._loss_scale() > 1:
      with self:  # re-enter the gradient tape so it sees the loss scaling
        loss_scale = self._loss_scale()
        scaled_target = nest.map_structure(lambda t: t * loss_scale, target)

      old_grads = super(LossScaleGradientTape, self).gradient(
          scaled_target, sources, output_gradients, unconnected_gradients)
      inv_loss_scale = 1.0 / self._loss_scale()
      grads = nest.map_structure(lambda g: inv_loss_scale * g, old_grads)
      # Check for non-finite gradients possibly resulting from scaling
      _, ready_to_update = self._loss_scale.update(grads)

    if not self._outer_persistent:
      self._tape = None  # free up resources if a persistent tape was not needed
    return grads
