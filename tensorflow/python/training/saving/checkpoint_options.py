# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Options for saving Checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export


@tf_export("train.CheckpointOptions")
class CheckpointOptions(object):
  """Options for constructing a Checkpoint.

  Used as the `_options` argument to the `tf.Checkpoint` constructor to adjust
  how variables are saved.

  Example: Run IO ops on "localhost" while saving a checkpoint:

  ```
  step = tf.Variable(0, name="step")
  checkpoint = tf.Checkpoint(step=step)
  options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
  checkpoint.save("/tmp/ckpt", options=options)
  ```
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = ("experimental_io_device",)

  def __init__(self, experimental_io_device=None):
    """Creates an object that stores options for a Checkpoint.

    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.

        This is for example useful if you want to save to a local directory,
        such as "/tmp" when running in a distributed setting. In that case pass
        a device for the host where the "/tmp" directory is accessible.
    """
    self.experimental_io_device = experimental_io_device
