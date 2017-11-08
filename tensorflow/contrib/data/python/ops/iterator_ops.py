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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import gen_dataset_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import saver


def make_saveable_from_iterator(iterator):
  """Returns a SaveableObject for saving/restore iterator state using Saver.

  Args:
    iterator: Iterator.

  For example:

  ```python
  with tf.Graph().as_default():
    ds = tf.data.Dataset.range(10)
    iterator = ds.make_initializable_iterator()
    # Build the iterator SaveableObject.
    saveable_obj = tf.contrib.data.make_saveable_from_iterator(iterator)
    # Add the SaveableObject to the SAVEABLE_OBJECTS collection so
    # it can be automatically saved using Saver.
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
    saver = tf.train.Saver()

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
  return _Saveable(iterator._iterator_resource)  # pylint: disable=protected-access


class _Saveable(saver.BaseSaverBuilder.SaveableObject):
  """SaveableObject for saving/restoring iterator state."""

  def __init__(self, iterator_resource):
    serialized_iterator = gen_dataset_ops.serialize_iterator(iterator_resource)
    specs = [
        saver.BaseSaverBuilder.SaveSpec(serialized_iterator, "",
                                        iterator_resource.name + "-state")
    ]
    super(_Saveable, self).__init__(iterator_resource, specs,
                                    iterator_resource.name)

  def restore(self, restored_tensors, unused_restored_shapes):
    with ops.colocate_with(self.op):
      return gen_dataset_ops.deserialize_iterator(self.op, restored_tensors[0])
