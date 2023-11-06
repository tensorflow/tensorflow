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
"""Gradients for lookup operations."""

from tensorflow.python.framework import ops


ops.NotDifferentiable("LookupTableFind")
ops.NotDifferentiable("LookupTableFindV2")
ops.NotDifferentiable("LookupTableInsert")
ops.NotDifferentiable("LookupTableInsertV2")
ops.NotDifferentiable("LookupTableSize")
ops.NotDifferentiable("LookupTableSizeV2")
ops.NotDifferentiable("HashTable")
ops.NotDifferentiable("HashTableV2")
ops.NotDifferentiable("InitializeTable")
ops.NotDifferentiable("InitializeTableV2")
ops.NotDifferentiable("InitializeTableFromTextFile")
ops.NotDifferentiable("InitializeTableFromTextFileV2")
ops.NotDifferentiable("MutableDenseHashTable")
ops.NotDifferentiable("MutableDenseHashTableV2")
ops.NotDifferentiable("MutableHashTable")
ops.NotDifferentiable("MutableHashTableV2")
ops.NotDifferentiable("MutableHashTableOfTensors")
ops.NotDifferentiable("MutableHashTableOfTensorsV2")
