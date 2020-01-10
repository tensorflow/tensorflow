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

#ifndef HLO_OPS_BASE
#define HLO_OPS_BASE

include "mlir/IR/OpBase.td"

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

class BASE_HLO_ClzOp {
  string summary = "Count-leading-zeros (Clz) operator";

  string description = [{
    Returns the number of leading zeros in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_ComplexOp {
  string summary = "Complex operator";

  string description = [{
    Performs element-wise conversion of a pair of real and imaginary values to
    a complex value.
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

class BASE_HLO_Expm1Op {
  string summary = "Exponential minus one operator";

  string description = [{
    Returns `e^(operand) - 1` element-wise.

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

class BASE_HLO_GetDimensionSizeOp {
  string summary = "GetDimensionSize operator";

  string description = [{
    Returns the size of the given dimension of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#getdimensionsize.
  }];
}

class BASE_HLO_ImagOp {
  string summary = "Imag operator";

  string description = [{
    Returns `Imag(operand)` element-wise.
  }];
}

class BASE_HLO_IsFiniteOp {
  string summary = "IsFinite operator";

  string description = [{
    Tests whether each element of operand is finite, i.e., is not positive or
    negative infinity, and is not NaN. Returns a tensor of 1-bit integers with
    the same shape as the input, where each element is nonzero (i.e. true) if
    and only if the corresponding input element is finite.

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

class BASE_HLO_Log1pOp {
  string summary = "Log1p operator";

  string description = [{
    Returns `log(operand+1)` element-wise.

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

class BASE_HLO_NotOp {
  string summary = "Not operator";

  string description = [{
    Returns `!operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_PopulationCountOp {
  string summary = "PopulationCount operator";

  string description = [{
    Returns the number of bits set in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_RealOp {
  string summary = "Real operator";

  string description = [{
    Returns `Real(operand)` element-wise.
  }];
}

class BASE_HLO_RoundOp {
  string summary = "Round operator";

  string description = [{
    Returns `Round(operand)` element-wise, rounding to nearest integer with
    half-way cases rounding away from zero.

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

class BASE_HLO_SinOp {
  string summary = "Sin operator";

  string description = [{
    Returns `Sin(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

class BASE_HLO_SqrtOp {
  string summary = "Square-root operator";

  string description = [{
    Returns `sqrt(operand)` element-wise.

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
class BASE_HLO_PowOp {
  string summary = "Power operator";

  string description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_RemOp {
  string summary = "Remainder operator";

  string description = [{
    Returns `lhs % rhs` element-wise.

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

class BASE_HLO_ShiftLeftOp {
  string summary = "Shift Left operator";

  string description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_ShiftRightArithmeticOp {
  string summary = "Shift right arithmetic operator";

  string description = [{
    Returns arithmetic `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_ShiftRightLogicalOp {
  string summary = "Shift right logical operator";

  string description = [{
    Returns logical `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

class BASE_HLO_Atan2Op {
  string summary = "Atan2 operator";

  string description = [{
    Returns `atan2(lhs/rhs)` element-wise.

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

class BASE_HLO_XorOp {
  string summary = "Logical xor";

  string description = [{
    Returns `lhs xor rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA parallelism related op definitions.
//===----------------------------------------------------------------------===//

class BASE_HLO_ReplicaIdOp {
  string summary = "ReplicaId operator";

  string description = [{
    Returns the unique ID (int32 scalar) of the replica.

    The unique ID of each replica is an unsigned integer in the interval [0, N),
    where N is the number of replicas. Since all the replicas are running the
    same program, a ReplicaId() call in the program will return a different
    value on each replica.

    See https://www.tensorflow.org/xla/operation_semantics#replicaid.
  }];
}


class BASE_HLO_AllReduceOp {
  string summary = "AllReduce operator";

  string description = [{
    Performs a custom reduction across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allreduce.
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
// Fast Fourier Transform Type enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA FftType proto enum.
def HLO_FFT_TYPE_FFT : StrEnumAttrCase<"FFT">;
def HLO_FFT_TYPE_IFFT : StrEnumAttrCase<"IFFT">;
def HLO_FFT_TYPE_RFFT : StrEnumAttrCase<"RFFT">;
def HLO_FFT_TYPE_IRFFT : StrEnumAttrCase<"IRFFT">;

def HLO_FftTypeAttr : StrEnumAttr<"FftType",
    "XLA fast fourier transform type.",
    [HLO_FFT_TYPE_FFT, HLO_FFT_TYPE_IFFT,
     HLO_FFT_TYPE_RFFT, HLO_FFT_TYPE_IRFFT]>;

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

class BASE_HLO_AllToAllOp {
  string summary = "AllToAll";

  string description = [{
    AllToAll is a collective operation that sends data from all cores to all
    cores. It has two phases:
    - The scatter phase. On each core, the operand is split into `split_count`
      number of blocks along the `split_dimensions`, and the blocks are
      scattered to all cores, e.g., the i-th block is send to the i-th core.
    - The gather phase. Each core concatenates the received blocks along the
      `concat_dimension`.

    The participating cores can be configured by:
    - replica_groups: each ReplicaGroup contains a list of replica id
      participating in the computation (replica id for the current replica can
      be retrieved using ReplicaId op). AllToAll will be applied within
      subgroups in the specified order. For example,
      `replica_groups` = {{1,2,3}, {4,5,0}} means that an AllToAll will be applied
      within replicas {1, 2, 3}, and in the gather phase, the received blocks
      will be concatenated in the same order of 1, 2, 3. Then, another AllToAll
      will be applied within replicas 4, 5, 0, and the concatenation order is
      also 4, 5, 0. If `replica_groups` is empty, all replicas belong to one
      group, in the concatenation order of their appearance.

    Prerequisites:
    - The dimension size of the operand on the split_dimension is divisible by
      `split_count`.
    - The operand's shape is not tuple.

    See https://www.tensorflow.org/xla/operation_semantics#alltoall
  }];
}

class BASE_HLO_BatchNormGradOp {
  string summary = "Batch Normalization Gradient";

  string description = [{
    Calculates gradients of batch norm.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
  }];
}

class BASE_HLO_BatchNormInferenceOp {
  string summary = "Batch Normalization for Inference";

  string description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];
}

class BASE_HLO_BatchNormTrainingOp {
  string summary = "Batch Normalization for Training";

  string description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
  }];
}

class BASE_HLO_BitcastConvertOp {
  string summary = "BitcastConvert operator";

  string description = [{
    Similar to a 'tf.bitcast' in TensorFlow, performs an element-wise bitcast
    operation from a data shape to a target shape. The dimensions must match,
    and the conversion is an element-wise one. Bitcast is implemented as a
    low-level cast, so machines with different floating-point representations
    will give different results.

    See https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype.
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

class BASE_HLO_CholeskyOp {
  string summary = "Cholesky operator";

  string description = [{
  Computes the Cholesky decomposition of a batch of symmetric (Hermitian)
  positive definite matrices.

  If lower is true, computes lower-triangular matrices l such that
  `a=l.Transpose(l)`. If lower is false, computes upper-triangular matrices u such
  that `a=Transpose(u).u`.

  Input data is read only from the lower/upper triangle of a, depending on the
  value of lower. Values from the other triangle are ignored. Output data is
  returned in the same triangle; the values in the other triangle are
  implementation-defined and may be anything.

  If the rank of a is greater than 2, a is treated as a batch of matrices, where
  all except the minor 2 dimensions are batch dimensions.

  If a is not symmetric (Hermitian) positive definite, the result is
  implementation-defined.

    See https://www.tensorflow.org/xla/operation_semantics#cholesky.
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

class BASE_HLO_CollectivePermuteOp {
  string summary = "CollectivePermute operator";

  string description = [{
    CollectivePermute is a collective operation that sends and receives data
    cross replicas.
    Note that there are the following restrictions on the source_target_pair:
    - Any two pairs should not have the same target replica id, and they should
    not have the same source replica id.
    - If a replica id is not a target in any pair, then the output on that
    replica is a tensor consists of 0(s) with the same shape as the input.

    See https://www.tensorflow.org/xla/operation_semantics#collectivepermute.

  }];
}
class BASE_HLO_ConcatenateOp {
   string summary = "XLA's concatenate op";

   string description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];
}

class BASE_HLO_CrossReplicaSumOp {
   string summary = "Sums input across replicated instances.";

   string description = [{
     For each of the replica groups, operands of the group devices are summed
     so that each device has the sum.

     For example, suppose there are 8 TPU devices: `[A, B, C, D, E, F, G, H]`.
     Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
     and `B, D, F, H` as group 1. Thus we get the outputs:
     `[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.

     See https://www.tensorflow.org/xla/operation_semantics#crossreplicasum.
   }];
}

class BASE_HLO_ConvOp {
  string summary = "Convolution operator";

  string description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];
}

class BASE_HLO_CustomCallOp {
  string summary = "CustomCall operator";

  string description = [{
    A custom call invokes code external to XLA. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.
    `backend_config` can encode arbitrarily large amounts of information.

    See https://www.tensorflow.org/xla/operation_semantics#customcall.
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

class BASE_HLO_FftOp {
  string summary = "Fast fourier transform operator";

  string description = [{
    Returns the fast-fourier-transform of the input array.

    See
    https://www.tensorflow.org/xla/operation_semantics#fft.
  }];
}

class BASE_HLO_GatherOp{
  string summary = "Gather operator";

  string description = [{
    Stitches together several slices of an input array.

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];
}

class BASE_HLO_MapOp {
  string summary = "Map operator";

  string description = [{
  Applies a scalar function over the given operands arrays, producing an array
  of the same dimensions where each element is the result of the mapped function
  applied to the corresponding elements in the input arrays.

  The mapped function is an arbitrary computation with the restriction that it
  has N inputs of scalar type T and a single output with type S. The output has
  the same dimensions as the operands except that the element type T is replaced
  with S.

  See https://www.tensorflow.org/xla/operation_semantics#map.
  }];
}

class BASE_HLO_ReshapeOp {
  string summary = "Reshape operator";

  string description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];
}

class BASE_HLO_ScatterOp {
  string summary = "Scatter operator";

  string description = [{
    Generates a result which is the value of the input array `operand`,
    with several slices (at indices specified by `scatter_indices`)
    updated with the values in `updates` using `update_computation`.

    See https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];
}

class BASE_HLO_SelectOp {
  string summary = "Select operator";

  string description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`.

    `pred`, `on_true` and `on_false` must be broadcast compatible.
  }];
}

class BASE_HLO_SelectAndScatterOp {
  string summary = "SelectAndScatter operator";

  string description = [{
    Runs a windowed selection `select` function over `operand` with shape
    `window_dimensions` and stride `window_strides`. This will produce an amount
    of selected locations whose shape matches `source`. These are then scattered
    to the output which is initialized with `init_value`.
    Multiple scattered elements which land in the same output location are
    combined using the `scatter` function.

    See https://www.tensorflow.org/xla/operation_semantics#selectandscatter.
  }];
}

class BASE_HLO_SetDimensionSizeOp {
  string summary = "SetDimensionSize operator";

  string description = [{
    Sets the dynamic size of operand's given dimension. Pass through the operand
    as result, with dynamic dimension tracked by the compiler. Padded values
    will be ignored by downstream reduction ops.

    See https://www.tensorflow.org/xla/operation_semantics#setdimensionsize.
  }];
}

class BASE_HLO_SortOp {
  string summary = "Sort operator";

  string description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
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

class BASE_HLO_TraceOp {
  string summary = "Trace operator";

  string description = [{
    Emits a logging message `tag` with the `operand`.
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
    Constructs an output of a given shape with random numbers generated
    following the uniform distribution over the interval `[a,b)`. The parameters
    and output element type have to be a boolean type, an integral type or a
    floating point types, and the types have to be consistent.

    See https://www.tensorflow.org/xla/operation_semantics#rnguniform.
  }];
}

class BASE_HLO_RngNormalOp {
  string summary = "RNG with normal distribution.";

  string description = [{
    Constructs an output of a given shape with random numbers generated
    following the normal distribution with parameters `mu` and `sigma`. The
    parameters and output shape have to have a floating point elemental type.
    The parameters furthermore have to be scalar valued.

    See https://www.tensorflow.org/xla/operation_semantics#rngnormal.
  }];
}

#endif // HLO_OPS_BASE
