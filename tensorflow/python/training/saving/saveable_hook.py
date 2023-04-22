# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""SaveableHook, for running callbacks at save and restore time."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.training.tracking import base


class SaveableHook(base.NoRestoreSaveable):
  """Base class for running callbacks at Save/Restore time.

  Subclasses should override one or both methods to modify or read variables
  during the saving process. No guarantees are made regarding the precedence
  of execution between multiple `SaveableHook` objects, but execution is
  guaranteed to occur before or after the respective event.

  Users should emit the SaveableHook alongside other SaveableObjects, such as
  in Trackable._gather_saveables_for_checkpoint().

  Saves a single constant in order to be compliant with the SaveableObject API.
  """

  def __init__(self, name):
    """Creates a `SaveableHook` object.

    Args:
      name: the name to save the object under.
    """
    super(SaveableHook, self).__init__(
        tensor=constant_op.constant(0),
        name=name,
    )

  @property
  def device(self):
    return self.op.device

  def before_save(self):
    """This method will be called before iterating devices for saving."""
    pass

  def after_restore(self):
    """This method will be called after each device is restored."""
    pass
