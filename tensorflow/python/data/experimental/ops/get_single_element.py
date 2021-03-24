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
  were converted to a `tf.data.Dataset`. Next, all the raw_features were
  mapped using the `preprocessing_fn` and the processed features were
  grouped into a single batch. The final `dataset` contains only one element
  which is a batch of all the processed features.

  NOTE: The `dataset` should contain only one element. Preferrably, a batch
    of processed features.

  Now, instead of creating an iterator for the `dataset` and retrieving the
  batch of features, the `tf.data.get_single_element()` function is used
  to skip the iterator creation process and directly output the batch of
  features.

  This can be particularly useful when your tensor transformations are
  expressed as `tf.data.Dataset` operations, and you want to use those
  transformations at serving time. Especially, when it comes to serving
  and inferecing using [estimator](https://www.tensorflow.org/guide/estimator)
  models.

  In order to export the saved estimator model, you need to generally define a
  `serving_input_fn` which would require the features to be processed by the
  model while inferencing.

  For example:

  ```python
  def serving_input_fn():
    # The function transforms the raw features of the data that is fed to
    # the model. Generally, an input_fn that expects data to be fed to the
    # model at serving time is used to get the raw_features and the tensor
    # placeholders.
    #
    # Reference: https://www.tensorflow.org/tfx/tutorials/transform/census

    input_fn = ... # estimator based data receiver function
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
        processed_features, serving_input_receiver.receiver_tensors
        # The `serving_input_receiver.receiver_tensors` are placeholder
        # for the `processed_features`
        )

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
