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
"""Ragged Tensors.

This package defines ops for manipulating ragged tensors (`tf.RaggedTensor`),
which are tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the `tf.RaggedTensor` class documentation
and the [Ragged Tensor Guide](/guide/ragged_tensor).

API docstring: tensorflow.ragged
"""
