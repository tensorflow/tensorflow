# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""Classes and functions for building TensorFlow graphs.

## Core graph data structures

@@Graph
@@Operation
@@Tensor

## Tensor types

@@DType
@@as_dtype

## Utility functions

@@device
@@name_scope
@@control_dependencies
@@convert_to_tensor
@@get_default_graph
@@import_graph_def

## Graph collections

@@add_to_collection
@@get_collection
@@GraphKeys

## Defining new operations

@@RegisterGradient
@@NoGradient
@@RegisterShape
@@TensorShape
@@Dimension
@@op_scope
@@get_seed
"""

# Classes used when building a Graph.
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.ops import SparseTensor
from tensorflow.python.framework.ops import SparseTensorValue
from tensorflow.python.framework.ops import IndexedSlices

# Utilities used when building a Graph.
from tensorflow.python.framework.ops import device
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.framework.ops import op_scope
from tensorflow.python.framework.ops import control_dependencies
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import GraphKeys
from tensorflow.python.framework.ops import add_to_collection
from tensorflow.python.framework.ops import get_collection
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.random_seed import get_seed
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.python.framework.importer import import_graph_def

# Needed when you defined a new Op in C++.
from tensorflow.python.framework.ops import RegisterGradient
from tensorflow.python.framework.ops import NoGradient
from tensorflow.python.framework.ops import RegisterShape
from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.framework.tensor_shape import TensorShape

from tensorflow.python.framework.types import *
