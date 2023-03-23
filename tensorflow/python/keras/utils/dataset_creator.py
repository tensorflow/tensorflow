# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Input dataset creator for `model.fit`."""

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.util.tf_export import keras_export



@keras_export('keras.utils.experimental.DatasetCreator', v1=[])
class DatasetCreator(object):
  """Object that returns a `tf.data.Dataset` upon invoking.

  `tf.keras.utils.experimental.DatasetCreator` is designated as a supported type
  for `x`, or the input, in `tf.keras.Model.fit`. Pass an instance of this class
  to `fit` when using a callable (with a `input_context` argument) that returns
  a `tf.data.Dataset`.

  ```python
  model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
  model.compile(tf.keras.optimizers.SGD(), loss="mse")

  def dataset_fn(input_context):
    global_batch_size = 64
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat()
    dataset = dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset

  input_options = tf.distribute.InputOptions(
      experimental_fetch_to_device=True,
      experimental_per_replica_buffer_size=2)
  model.fit(tf.keras.utils.experimental.DatasetCreator(
      dataset_fn, input_options=input_options), epochs=10, steps_per_epoch=10)
  ```

  `Model.fit` usage with `DatasetCreator` is intended to work across all
  `tf.distribute.Strategy`s, as long as `Strategy.scope` is used at model
  creation:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver)
  with strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
  model.compile(tf.keras.optimizers.SGD(), loss="mse")
  ...
  ```

  Note: When using `DatasetCreator`, `steps_per_epoch` argument in `Model.fit`
  must be provided as the cardinality of such input cannot be inferred.

  Args:
    dataset_fn: A callable that takes a single argument of type
      `tf.distribute.InputContext`, which is used for batch size calculation and
      cross-worker input pipeline sharding (if neither is needed, the
      `InputContext` parameter can be ignored in the `dataset_fn`), and returns
      a `tf.data.Dataset`.
    input_options: Optional `tf.distribute.InputOptions`, used for specific
      options when used with distribution, for example, whether to prefetch
      dataset elements to accelerator device memory or host device memory, and
      prefetch buffer size in the replica device memory. No effect if not used
      with distributed training. See `tf.distribute.InputOptions` for more
      information.
  """
  class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(
                names=['no', 'yes'],
                doc='Whether this is a picture of a cat'),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train': self._generate_examples(path=extracted_path / 'train_images'),
        'test': self._generate_examples(path=extracted_path / 'test_images'),
    }

  def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
    """Generator of examples for each split."""
    for img_path in path.glob('*.jpeg'):
      # Yields (key, example)
      yield img_path.name, {
          'image': img_path,
          'label': 'yes' if img_path.name.startswith('yes_') else 'no',
      }

  def __init__(self, dataset_fn, input_options=None):
    if not callable(dataset_fn):
      raise TypeError('`dataset_fn` for `DatasetCreator` must be a `callable`.')
    if input_options and (not isinstance(input_options,
                                         distribute_lib.InputOptions)):
      raise TypeError('`input_options` for `DatasetCreator` must be a '
                      '`tf.distribute.InputOptions`.')

    self.dataset_fn = dataset_fn
    self.input_options = input_options

  def __call__(self, *args, **kwargs):
    # When a `DatasetCreator` is invoked, it forwards args/kwargs straight to
    # the callable.
    dataset = self.dataset_fn(*args, **kwargs)
    if not isinstance(dataset, data_types.DatasetV2):
      raise TypeError('The `callable` provided to `DatasetCreator` must return '
                      'a Dataset.')
    return dataset
