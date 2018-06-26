"""Dependency tracking for checkpointable objects."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training.checkpointable import base


class NoDependency(object):
  """Allows attribute assignment to `Checkpointable` objects with no dependency.

  Example usage:
  ```python
  obj = Checkpointable()
  obj.has_dependency = tf.Variable(0., name="dep")
  obj.no_dependency = NoDependency(tf.Variable(1., name="nodep"))
  assert obj.no_dependency.name == "nodep:0"
  ```

  `obj` in this example has a dependency on the variable "dep", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  """

  def __init__(self, value):
    self.value = value


class NotCheckpointable(object):
  """Marks instances of child classes as unsaveable using an object-based API.

  Useful for marking objects which would otherwise look checkpointable because
  of inheritance (e.g. through `Layer`) as not checkpointable. Inheriting from
  `NotCheckpointable` does not prevent an object from being assigned to any
  attributes, but will throw an error on save/restore.
  """
  pass


class Checkpointable(base.CheckpointableBase):
  """Manages dependencies on other objects.

  `Checkpointable` objects may have dependencies: other `Checkpointable` objects
  which should be saved if the object declaring the dependency is saved. A
  correctly saveable program has a dependency graph such that if changing a
  global variable affects an object (e.g. changes the behavior of any of its
  methods) then there is a chain of dependencies from the influenced object to
  the variable.

  Dependency edges have names, and are created implicitly when a
  `Checkpointable` object is assigned to an attribute of another
  `Checkpointable` object. For example:

  ```
  obj = Checkpointable()
  obj.v = ResourceVariable(0.)
  ```

  The `Checkpointable` object `obj` now has a dependency named "v" on a
  variable.

  `Checkpointable` objects may specify `Tensor`s to be saved and restored
  directly (e.g. a `Variable` indicating how to save itself) rather than through
  dependencies on other objects. See
  `Checkpointable._gather_saveables_for_checkpoint` for details.
  """

  def __setattr__(self, name, value):
    """Support self.foo = checkpointable syntax."""
    # Perform the attribute assignment, and potentially call other __setattr__
    # overrides such as that for tf.keras.Model.
    no_dependency = isinstance(value, NoDependency)
    if no_dependency:
      value = value.value
    super(Checkpointable, self).__setattr__(name, value)
    if not no_dependency and isinstance(value, base.CheckpointableBase):
      self._track_checkpointable(
          value, name=name,
          # Allow the user to switch the Checkpointable which is tracked by this
          # name, since assigning a new variable to an attribute has
          # historically been fine (e.g. Adam did this).
          # TODO(allenl): Should this be a warning once Checkpointable save/load
          # is usable?
          overwrite=True)
