# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for registering the saving/loading steps for advanced objects."""

from tensorflow.python.saved_model.registration.registration import get_registered_class
from tensorflow.python.saved_model.registration.registration import get_registered_class_name
from tensorflow.python.saved_model.registration.registration import get_registered_saver_name
from tensorflow.python.saved_model.registration.registration import get_restore_function
from tensorflow.python.saved_model.registration.registration import get_save_function
from tensorflow.python.saved_model.registration.registration import get_strict_predicate_restore

# These are currently an evolving feature. Use with care.
from tensorflow.python.saved_model.registration.registration import register_checkpoint_saver
from tensorflow.python.saved_model.registration.registration import register_serializable

from tensorflow.python.saved_model.registration.registration import RegisteredSaver
from tensorflow.python.saved_model.registration.registration import validate_restore_function


def register_tf_serializable(name=None, predicate=None):
  """See the docstring for `register_serializable`."""
  return register_serializable(package="tf", name=name, predicate=predicate)


def register_tf_checkpoint_saver(name=None,
                                 predicate=None,
                                 save_fn=None,
                                 restore_fn=None,
                                 strict_predicate_restore=True):
  """See the docstring for `register_checkpoint_saver`."""
  return register_checkpoint_saver(
      package="tf",
      name=name,
      predicate=predicate,
      save_fn=save_fn,
      restore_fn=restore_fn,
      strict_predicate_restore=strict_predicate_restore)
