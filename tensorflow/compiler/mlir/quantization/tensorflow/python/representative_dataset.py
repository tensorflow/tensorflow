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
