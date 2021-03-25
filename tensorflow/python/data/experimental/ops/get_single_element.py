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
"""Python wrappers for Datasets and Iterators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.get_single_element", "data.get_single_element")
@deprecation.deprecated_endpoints("data.experimental.get_single_element")
def get_single_element(dataset):
  """Returns the single element of the `dataset` as a nested structure of tensors.

  The function enables you to use a `tf.data.Dataset` in a stateless
  "tensor-in tensor-out" expression, without creating an iterator.
  This facilitates the ease of data transformation on tensors using the
  optimized `tf.data.Dataset` abstraction on top of them.

  For example, lets consider a `preprocessing_fn` which would take as an
  input the raw features and returns the processed feature along with
  it's label.

  ```python
  def preprocessing_fn(raw_feature):
    # ... the raw_feature is preprocessed as per the use-case
    return feature

  raw_features = ...  # input batch of BATCH_SIZE elements.
  dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
             .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
             .batch(BATCH_SIZE))

  processed_features = tf.data.get_single_element(dataset)
  ```

  In the above example, the `raw_features` tensor of length=BATCH_SIZE
  was converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was
  mapped using the `preprocessing_fn` and the processed features were
  grouped into a single batch. The final `dataset` contains only one element
  which is a batch of all the processed features.

  NOTE: The `dataset` should contain only one element.

  Now, instead of creating an iterator for the `dataset` and retrieving the
  batch of features, the `tf.data.get_single_element()` function is used
  to skip the iterator creation process and directly output the batch of
  features.

  This can be particularly useful when your tensor transformations are
  expressed as `tf.data.Dataset` operations, and you want to use those
  transformations while serving your model. Especially, when it comes to
  serving `tf.keras` models using a `SavedModel`.

  tf.keras
  --------

  ```python
  import tensorflow as tf

  # Load data
  (x_train, y_train), (x_test, y_test) = (tf.keras.datasets.
                                        fashion_mnist.load_data())
  CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
    "Shirt", "Sneaker", "Bag", "Ankle Boot"
  ]

  def make_dataset(images, labels):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.shuffle(buffer_size=1024)
    ds = ds.map(lambda x, y: (x / 255, y))
    return ds.batch(batch_size=128)

  def make_model():
    # Add the layers
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
    ])
    # Compile the model
    model.compile( optimizer='adam', metrics=['accuracy'],
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    return model

  model = make_model()
  train_data = make_dataset(x_train, y_train)
  test_data = make_dataset(x_test, y_test)
  model.fit(train_data, validation_data=test_data, epochs=10)

  # There is a need to pre-process the data before inference as the model
  # will throw exceptions if the input is not of the desired shape. Thus, to
  # serialize and save the pre-processing functionality along with the model,
  # a `PreprocessingModel` is defined with a custom serving function called
  # `serving_fn`.

  class PreprocessingModel(tf.keras.Model):
    def __init__(self, model):
      super().__init__(self)
      self.model = model

    @tf.function(input_signature=[
      tf.TensorSpec([None, 28, 28],dtype=tf.uint8)
    ])
    def serving_fn(self, data):
      # Strictly speaking, you do not need to use `tf.data` here at all, as simple
      # `data / 255` would suffice. However, the point here is to illustrate that
      # `tf.data` can be leveraged for optimized processing operations.
      ds = tf.data.Dataset.from_tensor_slices(data)
      ds = ds.map(lambda x: x / 255, num_parallel_calls=16)
      ds = ds.batch(batch_size=16)
      probabilities = self.model(tf.data.experimental.get_single_element(ds))
      return tf.argmax(probabilities, axis=-1)

  preprocessing_model = PreprocessingModel(model)
  fmnist_save_path = "fmnist"

  tf.saved_model.save(preprocessing_model, fmnist_save_path,
                signatures={'serving_default': preprocessing_model.serving_fn})
  ```

  Estimators
  -----------

  In the case of estimators, you need to generally define a `serving_input_fn`
  which would require the features to be processed by the model while
  inferencing.

  ```python
  def serving_input_fn():
    # The function transforms the raw features of the data that is fed to
    # the model. Generally, an input_fn that expects data to be fed to the
    # model at serving time is used to get the raw_features and the tensor
    # placeholders.

    raw_feature_spec = ... # Spec for the raw_features
    input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    )
    # the `input_fn` is an estimator based data receiver function. The above
    # `tf.estimator.export.build_parsing_serving_input_receiver_fn()` function
    # expects features in the form of `tf.Example`s to be fed directly. Additionally,
    # API's such as `tf.estimator.export.build_raw_serving_input_receiver_fn()` are
    # also available which enables you to pass the features directly while serving.

    serving_input_receiver = input_fn()
    raw_features = serving_input_receiver.features

    def preprocessing_fn(raw_feature):
      # ... the raw_feature is preprocessed as per the use-case
      return feature

    dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
              .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
              .batch(BATCH_SIZE))

    processed_features = tf.data.get_single_element(dataset)

    return tf.estimator.export.ServingInputReceiver(
        processed_features, serving_input_receiver.receiver_tensors)

  estimator = ... # A pre-built or custom estimator
  estimator.export_saved_model(your_exported_model_dir, serving_input_fn)
  ```

  Args:
    dataset: A `tf.data.Dataset` object containing a single element.

  Returns:
    A nested structure of `tf.Tensor` objects, corresponding to the single
    element of `dataset`.

  Raises:
    TypeError: if `dataset` is not a `tf.data.Dataset` object.
    InvalidArgumentError (at runtime): if `dataset` does not contain exactly
      one element.
  """
  if not isinstance(dataset, dataset_ops.DatasetV2):
    raise TypeError("`dataset` must be a `tf.data.Dataset` object.")

  # pylint: disable=protected-access
  return structure.from_compatible_tensor_list(
      dataset.element_spec,
      gen_dataset_ops.dataset_to_single_element(
          dataset._variant_tensor, **dataset._flat_structure))  # pylint: disable=protected-access
