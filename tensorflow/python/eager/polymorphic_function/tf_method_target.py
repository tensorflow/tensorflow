# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Module for the TFMethodTarget Class."""

import weakref

from tensorflow.python.util import tf_inspect


# When a method is bound to objects of this type, it allows AutoGraph to
# recover a weak reference the original method's self pointer, so that it can
# execute it consistent with class_method_to_instance_method's
# bound_method_wrapper.
# TODO(b/119246461): This is not pretty. Use a descriptor instead?
class TfMethodTarget:
  """Binding target for methods replaced by function and defun."""

  __slots__ = ("weakrefself_target__", "weakrefself_func__")

  def __init__(self, target, original_python_function):
    self.weakrefself_target__ = target
    self.weakrefself_func__ = weakref.ref(original_python_function)

  @property
  def target(self):
    return self.weakrefself_target__()

  @property
  def target_class(self):
    true_self = self.weakrefself_target__()
    if tf_inspect.isclass(true_self):
      # Class method
      return true_self
    else:
      return true_self.__class__

  def call(self, args, kwargs):
    wrapped_fn = self.weakrefself_func__()
    return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)
