# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""A TensorSpec class."""

from tensorflow.python.framework import tensor

# This file is a pass-through to `tensor.py`. New references to this file
# should not be added - use `framework.tensor` directly instead.

# TODO(tristenallen) - Remove once existing references are updated.
DenseSpec = tensor.DenseSpec
TensorSpec = tensor.TensorSpec
BoundedTensorSpec = tensor.BoundedTensorSpec
