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
"""Defines types required for representative datasets for quantization."""

from typing import Iterable, Mapping, Union

from tensorflow.python.client import session
from tensorflow.python.types import core

# A representative sample is a map of: input_key -> input_value.
# Ex.: {'dense_input': tf.constant([1, 2, 3])}
# Ex.: {'x1': np.ndarray([4, 5, 6]}
RepresentativeSample = Mapping[str, core.TensorLike]

# A representative dataset is an iterable of representative samples.
RepresentativeDataset = Iterable[RepresentativeSample]

# A type representing a map from: signature key -> representative dataset.
# Ex.: {'serving_default': [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])],
#       'other_signature_key': [tf.constant([[2, 2], [9, 9]])]}
RepresentativeDatasetMapping = Mapping[str, RepresentativeDataset]

# A type alias expressing that it can be either a RepresentativeDataset or
# a mapping of signature key to RepresentativeDataset.
RepresentativeDatasetOrMapping = Union[RepresentativeDataset,
                                       RepresentativeDatasetMapping]


def replace_tensors_by_numpy_ndarrays(
    repr_ds: RepresentativeDataset,
    sess: session.Session) -> RepresentativeDataset:
  """Replaces tf.Tensors in samples by their evaluated numpy arrays.

  Note: This should be run in graph mode (default in TF1) only.

  Args:
    repr_ds: Representative dataset to replace the tf.Tensors with their
      evaluated values. `repr_ds` is iterated through, so it may not be reusable
      (e.g. if it is a generator object).
    sess: Session instance used to evaluate tf.Tensors.

  Returns:
    The new representative dataset where each tf.Tensor is replaced by its
    evaluated numpy ndarrays.
  """
  new_repr_ds = []
  for sample in repr_ds:
    new_sample = {}
    for input_key, input_data in sample.items():
      # Evaluate the Tensor to get the actual value.
      if isinstance(input_data, core.Tensor):
        input_data = input_data.eval(session=sess)

      new_sample[input_key] = input_data

    new_repr_ds.append(new_sample)
  return new_repr_ds
