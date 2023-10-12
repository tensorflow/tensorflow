# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Dataset types."""

import abc

from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.types.data.Dataset", v1=[])
class DatasetV2(abc.ABC):
  """Represents the TensorFlow 2 type `tf.data.Dataset`."""


@tf_export(v1=["__internal__.types.data.Dataset"])
class DatasetV1(DatasetV2, abc.ABC):
  """Represents the TensorFlow 1 type `tf.data.Dataset`."""
