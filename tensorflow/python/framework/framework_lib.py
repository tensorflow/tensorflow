# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=unused-import,g-bad-import-order
"""Classes and functions for building TensorFlow graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Classes used when building a Graph.
from tensorflow.python.framework.device import DeviceSpec
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import IndexedSlices

from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.framework.sparse_tensor import SparseTensorValue

# Utilities used when building a Graph.
from tensorflow.python.framework.ops import device
from tensorflow.python.framework.ops import container
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.ops import op_scope
from tensorflow.python.framework.ops import colocate_with
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import add_to_collection
from tensorflow.python.framework.ops import add_to_collections
from tensorflow.python.framework.ops import get_collection
from tensorflow.python.framework.ops import get_collection_ref
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.ops import convert_to_tensor_or_indexed_slices
from tensorflow.python.framework.random_seed import get_seed
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.framework.sparse_tensor import convert_to_tensor_or_sparse_tensor
from tensorflow.python.framework.importer import import_graph_def

# Utilities for working with Tensors
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray

# Needed when you defined a new Op in C++.
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import NotDifferentiable
from tensorflow.python.framework.ops import NoGradient
from tensorflow.python.framework.ops import RegisterShape
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.framework.tensor_shape import TensorShape

# Needed when interfacing tensorflow to new array libraries
from tensorflow.python.framework.ops import register_tensor_conversion_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework.dtypes import *  # pylint: disable=redefined-builtin

# Load a TensorFlow plugin
from tensorflow.python.framework.load_library import *
# pylint: enable=wildcard-import
