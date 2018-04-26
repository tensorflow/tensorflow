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
"""Implements placement strategies for cov and inv ops, cov variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.framework import ops as tf_ops


def _make_thunk_on_device(func, device):
  def thunk():
    with tf_ops.device(device):
      return func()
  return thunk


class RoundRobinPlacementMixin(object):
  """Implements round robin placement strategy for ops and variables."""

  def __init__(self, cov_devices=None, inv_devices=None, **kwargs):
    """Initializes the RoundRobinPlacementMixin class.

    Args:
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      **kwargs: Need something here?

    """
    super(RoundRobinPlacementMixin, self).__init__(**kwargs)
    self._cov_devices = cov_devices
    self._inv_devices = inv_devices

  def make_vars_and_create_op_thunks(self, scope=None):
    """Make vars and create op thunks w/ a round-robin device placement strat.

    For each factor, all of that factor's cov variables and their associated
    update ops will be placed on a particular device.  A new device is chosen
    for each factor by cycling through list of devices in the
    `self._cov_devices` attribute. If `self._cov_devices` is `Non`e then no
    explicit device placement occurs.

    An analogous strategy is followed for inverse update ops, with the list of
    devices being given by the `self._inv_devices` attribute.

    Inverse variables on the other hand are not placed on any specific device
    (they will just use the current the device placement context, whatever
    that happens to be).  The idea is that the inverse variable belong where
    they will be accessed most often, which is the device that actually applies
    the preconditioner to the gradient. The user will be responsible for setting
    the device context for this.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all thunks will execute, inside of a variable scope of the given
        name. (Default: None)

    Returns:
      cov_update_thunks: List of cov update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
      inv_update_thunks: List of inv update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
    """
    # Note: `create_ops_and_vars_thunks` is implemented in `FisherEstimator`.
    (cov_variable_thunks_raw, cov_update_thunks_raw, inv_variable_thunks_raw,
     inv_update_thunks_raw) = self.create_ops_and_vars_thunks(scope=scope)

    if self._cov_devices:
      cov_update_thunks = []
      for cov_variable_thunk, cov_update_thunk, device in zip(
          cov_variable_thunks_raw, cov_update_thunks_raw,
          itertools.cycle(self._cov_devices)):
        with tf_ops.device(device):
          cov_variable_thunk()
        cov_update_thunks.append(_make_thunk_on_device(cov_update_thunk,
                                                       device))
    else:
      for cov_variable_thunk in cov_variable_thunks_raw:
        cov_variable_thunk()
      cov_update_thunks = cov_update_thunks_raw

    for inv_variable_thunk in inv_variable_thunks_raw:
      inv_variable_thunk()

    if self._inv_devices:
      inv_update_thunks = []
      for inv_update_thunk, device in zip(inv_update_thunks_raw,
                                          itertools.cycle(self._inv_devices)):
        inv_update_thunks.append(_make_thunk_on_device(inv_update_thunk,
                                                       device))
    else:
      inv_update_thunks = inv_update_thunks_raw

    return cov_update_thunks, inv_update_thunks
