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
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.get_single_element()`.")
@tf_export("data.experimental.get_single_element")
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

  processed_features = tf.data.experimental.get_single_element(dataset)
  ```

  In the above example, the `raw_features` tensor of length=BATCH_SIZE
  was converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was
  mapped using the `preprocessing_fn` and the processed features were
  grouped into a single batch. The final `dataset` contains only one element
  which is a batch of all the processed features.

  NOTE: The `dataset` should contain only one element.

  Now, instead of creating an iterator for the `dataset` and retrieving the
  batch of features, the `tf.data.experimental.get_single_element()` function
  is used to skip the iterator creation process and directly output the batch
  of features.

  This can be particularly useful when your tensor transformations are
  expressed as `tf.data.Dataset` operations, and you want to use those
  transformations while serving your model.

  # Keras

  ```python

  model = ... # A pre-built or custom model

  class PreprocessingModel(tf.keras.Model):
    def __init__(self, model):
      super().__init__(self)
      self.model = model

    @tf.function(input_signature=[...])
    def serving_fn(self, data):
      ds = tf.data.Dataset.from_tensor_slices(data)
      ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
      ds = ds.batch(batch_size=BATCH_SIZE)
      return tf.argmax(
        self.model(tf.data.experimental.get_single_element(ds)),
        axis=-1
      )

  preprocessing_model = PreprocessingModel(model)
  your_exported_model_dir = ... # save the model to this path.
  tf.saved_model.save(preprocessing_model, your_exported_model_dir,
                signatures={'serving_default': preprocessing_model.serving_fn})
  ```

  # Estimator

  In the case of estimators, you need to generally define a `serving_input_fn`
  which would require the features to be processed by the model while
  inferencing.

  ```python
  def serving_input_fn():

    raw_feature_spec = ... # Spec for the raw_features
    input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    )
    serving_input_receiver = input_fn()
    raw_features = serving_input_receiver.features

    def preprocessing_fn(raw_feature):
      # ... the raw_feature is preprocessed as per the use-case
      return feature

    dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
              .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
              .batch(BATCH_SIZE))

    processed_features = tf.data.experimental.get_single_element(dataset)

    # Please note that the value of `BATCH_SIZE` should be equal to
    # the size of the leading dimension of `raw_features`. This ensures
    # that `dataset` has only element, which is a pre-requisite for
    # using `tf.data.experimental.get_single_element(dataset)`.

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
    InvalidArgumentError: (at runtime) if `dataset` does not contain exactly
      one element.
  """
  if not isinstance(dataset, dataset_ops.DatasetV2):
    raise TypeError(
        f"Invalid `dataset`. Expected a `tf.data.Dataset` object "
        f"but got {type(dataset)}.")

  return dataset.get_single_element()
