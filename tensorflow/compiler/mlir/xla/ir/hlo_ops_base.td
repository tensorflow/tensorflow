/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef HLO_OPS_BASE
#else
#define HLO_OPS_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def HLO_Int : IntOfWidths<[8, 16, 32, 64]>;
def HLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

class BASE_HLO_ConstOp {
  string summary = "Constant operator";

  string description = [{
    Represents a constant value.
  }];
}

class BASE_HLO_IotaOp {
  string summary = "Iota operator";

  string description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.
  }];
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class BASE_HLO_AbsOp {
  string summary = "Absolute value operator";

  string description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_CeilOp {
  string summary = "Ceil operator";

  string description = [{
    Returns `Ceil(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_ConvertOp {
  string summary = "Convert operator";

  string description = [{
    Performs element-wise conversion of values from one type to another, e.g.
    float to int.

    See https://www.tensorflow.org/xla/operation_semantics#convertelementtype.
  }];
}

class BASE_HLO_CosOp {
  string summary = "Cos operator";

  string description = [{
    Returns `Cos(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_ExpOp {
  string summary = "Exponential operator";

  string description = [{
    Returns `e^(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_FloorOp {
  string summary = "Floor operator";

  string description = [{
    Returns `Floor(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_LogOp {
  string summary = "Logarithm operator";

  string description = [{
    Returns `log(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_NegOp {
  string summary = "Negation operator";

  string description = [{
    Returns `-operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_RsqrtOp {
  string summary = "Reciprocal Square-root operator";

  string description = [{
    Returns `1.0 / sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_SignOp {
  string summary = "Sign operator";

  string description = [{
    Returns `sign(operand)` element-wise, where

    ```
    sign(x) = -1  : x < 0
            = -0  : x = -0
            = NaN : x = NaN
            = +0  : x = +0
            = 1   : x > 0
    ```

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_TanhOp {
  string summary = "Tanh operator";

  string description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
def BroadcastDimAttr : OptionalAttr<I64ElementsAttr>;

class BASE_HLO_AddOp {
  string summary = "Addition operator";

  string description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_DivOp {
  string summary = "Division operator";

  string description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_MaxOp {
  string summary = "Maximum operator";

  string description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_MinOp {
  string summary = "Minimum operator";

  string description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_MulOp {
  string summary = "Multiplication operator";

  string description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_SubOp {
  string summary = "Subtraction operator";

  string description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_AndOp {
  string summary = "Logical and";

  string description = [{
    Returns `lhs /\ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_OrOp {
  string summary = "Logical or";

  string description = [{
    Returns `lhs \/ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_ReduceOp {
  string summary = "Reduce operator";

  string description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];
}

class BASE_HLO_ReduceWindowOp {
  string summary = "ReduceWindow operator";

  string description = [{
    Returns the result of executing a reduction function over all elements in
    each window of one or more arrays in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reducewindow.
  }];
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//
class BASE_HLO_GetTupleElementOp {
  string summary = "GetTupleElement operator";

  string description = [{
    Returns a member of a tuple specified by an index.

    See https://www.tensorflow.org/xla/operation_semantics#gettupleelement.
  }];
}

class BASE_HLO_TupleOp {
   string summary = "XLA's tuple op";

   string description = [{
     Groups a set of tensor inputs into a single tuple object.

     See https://www.tensorflow.org/xla/operation_semantics#tuple.
   }];
}

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def HLO_PRECISION_DEFAULT : StrEnumAttrCase<"DEFAULT">;
def HLO_PRECISION_HIGH    : StrEnumAttrCase<"HIGH">;
def HLO_PRECISION_HIGHEST : StrEnumAttrCase<"HIGHEST">;

def HLO_PrecisionAttr : StrEnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [HLO_PRECISION_DEFAULT,  HLO_PRECISION_HIGH, HLO_PRECISION_HIGHEST]>;

// TODO(b/129153247) See if it's possible to also validate the size.
def HLO_PrecisionConfigAttr:
    OptionalAttr<
          TypedArrayAttrBase<HLO_PrecisionAttr, "Precision Config attribute">>;

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def HLO_COMPARISON_DIRECTION_EQ : StrEnumAttrCase<"EQ">;
def HLO_COMPARISON_DIRECTION_NE : StrEnumAttrCase<"NE">;
def HLO_COMPARISON_DIRECTION_GE : StrEnumAttrCase<"GE">;
def HLO_COMPARISON_DIRECTION_GT : StrEnumAttrCase<"GT">;
def HLO_COMPARISON_DIRECTION_LE : StrEnumAttrCase<"LE">;
def HLO_COMPARISON_DIRECTION_LT : StrEnumAttrCase<"LT">;

def HLO_ComparisonDirectionAttr : StrEnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      HLO_COMPARISON_DIRECTION_EQ,
      HLO_COMPARISON_DIRECTION_NE,
      HLO_COMPARISON_DIRECTION_GE,
      HLO_COMPARISON_DIRECTION_GT,
      HLO_COMPARISON_DIRECTION_LE,
      HLO_COMPARISON_DIRECTION_LT
    ]>;

class BASE_HLO_CompareOp {
  string summary = "Comparison operator";

  string description = [{
    Compares `lhs` and `rhs` elementwise according to `comparison_direction`.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

class BASE_HLO_SliceOp {
  string summary = "Slice operator";

  string description = [{
    Slices a portion of the `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#slice.
  }];
}

class BASE_HLO_DynamicSliceOp {
  string summary = "Dynamic Slice operator";

  string description = [{
    Extracts a sub-array from the input array at dynamic start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicslice.
  }];
}

class BASE_HLO_DynamicUpdateSliceOp {
  string summary = "Dynamic Update Slice operator";

  string description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];
}

//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

class BASE_HLO_BatchNormInferenceOp {
  string summary = "Batch Normalization for Inference";

  string description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];
}

class BASE_HLO_BroadcastOp  {
  string summary = "Broadcast a tensor to a higher rank by prepending dimensions";

  string description = [{
    Broadcasts the operand tensor to a higher rank by prepending
    `broadcast_sizes` to the dimensions. The current values of the operand are
    copied into the other dimensions.

    This is a more limited form of broadcasting, that corresponds to the XLA
    client Broadcast method. For a more general form of broadcasting, see the
    BroadcastInDimOp.

    See https://www.tensorflow.org/xla/operation_semantics#broadcast.
  }];
}

class BASE_HLO_BroadcastInDimOp  {
  string summary = "Broadcast a tensor into the given shape by adding dimensions.";

  string description = [{
    Broadcasts the `operand` tensor to a higher rank. This is not the limited
    form of broadcasting exposed as the XLA client broadcast op, but rather the
    more powerful "InDim" broadcasting, which is closer to the HLO broadcast op
    and exposed in the XLA client BroadcastInDim method.

    `broadcast_dimensions` maps the operand dimension number to the target shape
    dimension number. It must have the same size as the rank of the operand. The
    mapped dimensions must either be the same size or the dimension being
    broadcast from must be size 1 (degenerate broadcasting).

    For a scalar (0D tensor) operand, `broadcast_dimensions` must be empty. The
    The scalar value will be broadcast to every element in the target shape.

    See https://www.tensorflow.org/xla/broadcasting.
  }];
}

class BASE_HLO_ClampOp  {
  string summary = "Clamp operator";

  string description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.
  }];
}

class BASE_HLO_ConcatenateOp {
   string summary = "XLA's concantenate op";

   string description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];
}

class BASE_HLO_ConvOp {
  string summary = "Convolution operator";

  string description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];
}

class BASE_HLO_DotOp {
  string summary = "Dot operator";
  string description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
}

class BASE_HLO_DotGeneralOp {
  string summary = "General Dot operator";
  string description = [{
    Performs general dot products between vectors, vector/matrix and
    matrix/matrix multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dotgeneral.
  }];
}


class BASE_HLO_GatherOp{
  string summary = "Gather operator";

  string description = [{
    Stitches together several slices of an input array.

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];
}

class BASE_HLO_ReshapeOp {
  string summary = "Reshape operator";

  string description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];
}

class BASE_HLO_SelectOp {
  string summary = "Select operator";

  string description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`.

    `on_true` and `on_false` must be the same shape. For each element of `pred`,
    `res` has the corresponding element of `on_true` or `on_false` depending on
    the value in `pred`. `pred` must be the same shape as `on_true` and
    `on_false` or a scalar, in which case `res` is equal to either `on_true` or
    `on_false`.

    See https://www.tensorflow.org/xla/operation_semantics#select.
  }];
}

class BASE_HLO_ReverseOp {
  string summary = "Reverse operator";

  string description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];
}

class BASE_HLO_PadOp {
  string summary = "Pad operator";

  string description = [{
    Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration.

    See https://www.tensorflow.org/xla/operation_semantics#pad.
  }];
}

class BASE_HLO_TransposeOp {
  string summary = "Transpose operator";

  string description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];
}

class BASE_HLO_RngUniformOp {
  string summary = "RNG with uniform distribution.";

  string description = [{
    Constructs an output of a given shape with random numbers generated following
    the uniform distribution over the interval `[a,b)`.

    See https://www.tensorflow.org/xla/operation_semantics#rnguniform.
  }];
}

#endif // HLO_OPS_BASE
