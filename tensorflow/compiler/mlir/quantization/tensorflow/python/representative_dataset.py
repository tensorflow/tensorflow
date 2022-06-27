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

from typing import Iterable, Mapping, Tuple, Union

from tensorflow.python.types import core

# A representative sample should be either:
# 1. (signature_key, {input_key -> input_value}) tuple, or
# 2. {input_key -> input_value} mappings.
RepresentativeSample = Union[Tuple[str, Mapping[str, core.TensorLike]],
                             Mapping[str, core.TensorLike]]

# A representative dataset is an iterable of representative samples.
RepresentativeDataset = Iterable[RepresentativeSample]
