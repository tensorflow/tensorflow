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
"""Options for saving SavedModels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util.tf_export import tf_export


@tf_export("saved_model.LoadOptions", v1=[])
class LoadOptions(object):
  """Options for loading a SavedModel.

  This function may be used in the `options` argument in functions that
  load a SavedModel (`tf.saved_model.load`, `tf.keras.models.load_model`).
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = ("experimental_io_device",)

  def __init__(self,
               experimental_io_device=None):
    """Creates an object that stores options for SavedModel loading.

    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.
        This is for example useful if you want to load from a local directory,
        such as "/tmp" when running in a distributed setting. In that case
        pass a device for the host where the "/tmp" directory is accessible.

    Example:

      load_options = tf.saved_model.LoadOptions(experimental_io_device=
        '/job:localhost')
      restoredmodel = tf.keras.models.load_model(saved_model_path,
                                                 options=load_options)

    """
    self.experimental_io_device = experimental_io_device
