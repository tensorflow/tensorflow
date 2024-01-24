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
"""Iterator ops."""

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


def _convert_external_state_policy_to_enum(external_state_policy):
  if isinstance(external_state_policy, options_lib.ExternalStatePolicy):
    return external_state_policy
  if external_state_policy == "warn":
    return options_lib.ExternalStatePolicy.WARN
  if external_state_policy == "ignore":
    return options_lib.ExternalStatePolicy.IGNORE
  if external_state_policy == "fail":
    return options_lib.ExternalStatePolicy.FAIL
  raise ValueError(
      f"Invalid `ExternalStatePolicy.` Supported values include 'warn', "
      f"'ignore', and 'fail.' Received {external_state_policy}."
  )


@tf_export("data.experimental.make_saveable_from_iterator")
@deprecation.deprecated(
    None, "`make_saveable_from_iterator` is intended for use in TF1 with "
    "`tf.compat.v1.Saver`. In TF2, use `tf.train.Checkpoint` instead.")
def make_saveable_from_iterator(iterator, external_state_policy=None):
  """Returns a SaveableObject for saving/restoring iterator state using Saver.

  Args:
    iterator: Iterator.
    external_state_policy: A string that identifies how to handle input
      pipelines that depend on external state. Possible values are
      'ignore': The external state is silently ignored.
      'warn': The external state is ignored, logging a warning.
      'fail': The operation fails upon encountering external state.
      By default we set it to 'fail'.

  Returns:
    A SaveableObject for saving/restoring iterator state using Saver.

  Raises:
    ValueError: If iterator does not support checkpointing.
    ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or
      'fail'.

  For example:

  ```python
  with tf.Graph().as_default():
    ds = tf.data.Dataset.range(10)
    iterator = ds.make_initializable_iterator()
    # Build the iterator SaveableObject.
    saveable_obj = tf.data.experimental.make_saveable_from_iterator(iterator)
    # Add the SaveableObject to the SAVEABLE_OBJECTS collection so
    # it can be automatically saved using Saver.
    tf.compat.v1.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
    saver = tf.compat.v1.train.Saver()

    while continue_training:
      ... Perform training ...
      if should_save_checkpoint:
        saver.save()
  ```

  Note: When restoring the iterator, the existing iterator state is completely
  discarded. This means that any changes you may have made to the Dataset
  graph will be discarded as well! This includes the new Dataset graph
  that you may have built during validation. So, while running validation,
  make sure to run the initializer for the validation input pipeline after
  restoring the checkpoint.

  Note: Not all iterators support checkpointing yet. Attempting to save the
  state of an unsupported iterator will throw an error.
  """
  if external_state_policy is None:
    external_state_policy = "fail"
  policy_enum = _convert_external_state_policy_to_enum(external_state_policy)
  return iterator_ops._IteratorSaveable(  # pylint: disable=protected-access
      iterator._iterator_resource,  # pylint: disable=protected-access
      iterator._iterator_resource.name,  # pylint: disable=protected-access
      external_state_policy=policy_enum)
