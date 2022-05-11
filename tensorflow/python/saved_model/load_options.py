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

from tensorflow.python.saved_model import save_options
from tensorflow.python.util.tf_export import tf_export


@tf_export("saved_model.LoadOptions", v1=[])
class LoadOptions(object):
  """Options for loading a SavedModel.

  This function may be used in the `options` argument in functions that
  load a SavedModel (`tf.saved_model.load`, `tf.keras.models.load_model`).
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = ("allow_partial_checkpoint", "experimental_io_device",
               "experimental_skip_checkpoint", "experimental_variable_policy")

  def __init__(self,
               allow_partial_checkpoint=False,
               experimental_io_device=None,
               experimental_skip_checkpoint=False,
               experimental_variable_policy=None):
    """Creates an object that stores options for SavedModel loading.

    *When to set `allow_partial_checkpoint=True`?*

    This can be used when loading a Keras model (`tf.keras.models.load_model`)
    with custom objects. When new variables are added to the custom object
    class, loading will fail the assertion check that all loaded variables have
    been restored, because the SavedModel checkpoint only contains the variables
    that were in original the custom object.
    See the following example:

    ```
    class Custom(tf.keras.Model):
      def __init__(self):
        super(Custom, self).__init__()
        self.v = tf.Variable(...)

      def call(self, inputs):
        return ...

    model = Custom()
    model.save(...)
    ```

    After saving, say that `Custom` is updated to include an additional
    variable.

    ```
    class Custom(tf.keras.Model):
      def __init__(self):
        super(Custom, self).__init__()
        self.v = tf.Variable(...)
        self.w = tf.Variable(...)

      def call(self, inputs):
        return ...
    ```

    `tf.keras.models.load_model(path, custom_objects={'Custom': Custom})` fails
    to load since `Custom.w` does not exist in the SavedModel checkpoint. To
    acknowledge that there are variables that are not restored from the
    checkpoint and successfully load the model, call:

    ```
    tf.keras.models.load_model(
      path, custom_objects={'Custom': Custom},
      options=tf.saved_model.LoadOptions(allow_partial_checkpoint=True))
    ```

    Args:
      allow_partial_checkpoint: bool. Defaults to `False`. When enabled, allows
        the SavedModel checkpoint to not entirely match the loaded object.
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.
        This is for example useful if you want to load from a local directory,
        such as "/tmp" when running in a distributed setting. In that case
        pass a device for the host where the "/tmp" directory is accessible.
      experimental_skip_checkpoint: bool. Defaults to `False`. If set to `True`,
        checkpoints will not be restored. Note that this in the majority of
        cases will generate an unusable model.
      experimental_variable_policy: string. The policy to apply to variables
        when loading. This is either a `saved_model.experimental.VariablePolicy`
        enum instance or one of its value strings (case is not important). See
        that enum documentation for details. A value of `None` corresponds to
        the default policy.

    Example:

      load_options = tf.saved_model.LoadOptions(experimental_io_device=
        '/job:localhost')
      restoredmodel = tf.keras.models.load_model(saved_model_path,
                                                 options=load_options)

    """
    self.experimental_io_device = experimental_io_device
    self.allow_partial_checkpoint = allow_partial_checkpoint
    self.experimental_skip_checkpoint = experimental_skip_checkpoint
    self.experimental_variable_policy = (
        save_options.VariablePolicy.from_obj(experimental_variable_policy))

