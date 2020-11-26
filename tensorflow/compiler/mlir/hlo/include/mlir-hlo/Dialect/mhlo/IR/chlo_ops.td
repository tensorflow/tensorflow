/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Defines "client" aligned HLO ops.
// These ops are not necessarily orthogonal or optimized for transformation but
// for ease of expression in certain cases deemed important for client
// libraries (i.e. implicit broadcasting, helper ops, etc).
// This dialect is considered to exist in addition to augment the mhlo
// dialect for ergonomic needs, not duplicate/replace it.
//
// The typical use of this dialect is for client libraries to be able to emit
// less constrained ops and rely on the conversion framework to lower any
// chlo ops to canonical mhlo ops.
//
// See: https://www.tensorflow.org/xla/operation_semantics

#ifndef CHLO_OPS
#define CHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"
include "mlir-hlo/Dialect/mhlo/IR/infer_fusibility_op_interface.td"

def HLOClient_Dialect : Dialect {
  let name = "chlo";
  let cppNamespace = "::mlir::chlo";
  let summary = [{
    Client HLO Ops
  }];

  let description = [{
    This dialect contains ops that align closely with the API surface area
    of the XlaBuilder C++ API, where such ops have semantics that go beyond
    what exists in the lower level dialects (such as `mhlo`). Essentially,
    whenever the client library uses syntactic sugar or composition
    of multiple ops for an API call, this dialect tries to model the API call
    and provide conversion patterns to fully materialize into lower level
    dialects.
  }];
}

class HLOClient_Op<string mnemonic, list<OpTrait> traits> :
    Op<HLOClient_Dialect, mnemonic, traits> {
  // TODO(b/129012527) Much of this custom verification should be expressed as
  // type constraints.
  let verifier = [{ return Verify(*this); }];
}

//===----------------------------------------------------------------------===//
// CHLO binary elementwise op definitions.
// From the client perspective, each of these support both explicit rank
// broadcasting (via the broadcast_dimensions attribute) and implicit degenerate
// shape broadcasting.
//
// These correspond to operations in the mhlo dialect without the
// "broadcast_" prefix, except that those ops require same-shaped operands and
// results.
//
// See:
//   https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
//   https://www.tensorflow.org/xla/broadcasting
//===----------------------------------------------------------------------===//

class HLOClient_BroadcastBinaryElementwiseOp<
  string mnemonic, list<OpTrait> traits> :
        HLOClient_Op<mnemonic,
            !listconcat(traits, [
              DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
              ["reifyReturnTypeShapes"]>])> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );

  let builders = [
    OpBuilderDAG<(ins "Value":$left, "Value":$right,
      "DenseIntElementsAttr":$broadcast_dimensions)>];

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type(results)
  }];
}

def HLOClient_BroadcastAddOp : HLOClient_BroadcastBinaryElementwiseOp<"broadcast_add",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Addition operator (with optional broadcasting)";

  string description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastAtan2Op : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_atan2",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Atan2 operator (with optional broadcasting)";

  string description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastDivOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_divide",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Division operator (with optional broadcasting)";

  string description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMaxOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_maximum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Maximum operator (with optional broadcasting)";

  string description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMinOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_minimum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Minimum operator (with optional broadcasting)";

  string description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMulOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_multiply",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Multiplication operator (with optional broadcasting)";

  string description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastPowOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_power",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Power operator (with optional broadcasting)";

  string description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastRemOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_remainder",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Remainder operator (with optional broadcasting)";

  string description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftLeftOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_left",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift left operator (with optional broadcasting)";

  string description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftRightArithmeticOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_arithmetic",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right arithmetic operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftRightLogicalOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_logical",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right logical operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastSubOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_subtract",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Subtraction operator (with optional broadcasting)";

  string description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
// The same description as the arithmetic binary elementwise ops applies.
//===----------------------------------------------------------------------===//

class HLOClient_BroadcastBinaryLogicalElementwiseOp<string mnemonic> :
    HLOClient_BroadcastBinaryElementwiseOp<
      mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def HLOClient_BroadcastAndOp: HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_and"> {
  string summary = "Logical and operator (with optional broadcasting)";

  string description = [{
    Returns `logical_and(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastOrOp: HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_or"> {
  string summary = "Logical or operator (with optional broadcasting)";

  string description = [{
    Returns `logical_or(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastXorOp : HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_xor"> {
  string summary = "Logical xor operator (with optional broadcasting)";

  string description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// Broadcasting complex op
//===----------------------------------------------------------------------===//

def HLOClient_BroadcastComplexOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_complex", [NoSideEffect]> {
  string summary = "Complex operator (with optional broadcasting)";

  string description = [{
    Performs element-wise conversion of a pair of real and imaginary values to
    a complex value.
  }];

  let arguments = (ins
    HLO_FpTensor:$lhs,
    HLO_FpTensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
  let results = (outs HLO_ComplexTensor);
}

//===----------------------------------------------------------------------===//
// Unary op
//===----------------------------------------------------------------------===//

class HLOClient_UnaryElementwiseOp<string mnemonic, list<OpTrait> traits,
    Type TensorType> : HLOClient_Op<mnemonic, !listconcat(traits, [
    InferFusibilityOpInterface, NoSideEffect, SameOperandsAndResultType])> {
  let arguments = (ins TensorType:$operand);
  let results = (outs TensorType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def HLOClient_AcosOp : HLOClient_UnaryElementwiseOp<"acos", [],
    HLO_FpOrComplexTensor> {
  let summary = "Acos operator";

  let description = [{
    Returns `Acos(operand)` element-wise.

    $$
    \acos(x) = 2 * \atan(\sqrt(1 - x^2) / (1 + x)) if x != -1
             = pi                                  if x == -1
    $$
  }];
}

def HLOClient_AtanOp : HLOClient_UnaryElementwiseOp<"atan", [],
    HLO_FpOrComplexTensor> {
  let summary = "Atan operator";

  let description = [{
    Returns `Atan(operand)` element-wise.

    $$
    \atan(x) = \atan2(x, 1)
    $$
  }];
}

def HLOClient_ConjOp : HLOClient_UnaryElementwiseOp<"conj", [],
    HLO_FpOrComplexTensor> {
  let summary = "Conj operator";

  let description = [{
    Returns `Conj(operand)` element-wise.

    $$
    \conj(x) = (\real(x), \neg(\imag(x)))
    $$
  }];
}

def HLOClient_SinhOp : HLOClient_UnaryElementwiseOp<"sinh", [],
    HLO_FpOrComplexTensor> {
  let summary = "Sinh operation";

  let description = [{
    Returns `Sinh(operand)` element-wise.

    $$
    \sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
             = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
    $$
  }];
}

def HLOClient_TanOp : HLOClient_UnaryElementwiseOp<"tan", [],
    HLO_FpOrComplexTensor> {
  let summary = "Tan operation";

  let description = [{
    Returns `Tan(operand)` element-wise.

    $$
    \tan(x) = \sin(x) / \cos(x)
    $$
  }];
}

def HLOClient_ConstantLikeOp : HLOClient_Op<"constant_like",
    [NoSideEffect, SameOperandsAndResultShape,
     InferTypeOpInterface,
     DeclareOpInterfaceMethods<InferShapedTypeOpInterface>,
     NativeOpTrait<"InferTensorType">]> {
  let summary = "Constant like operator";

  let description = [{
    Returns a splat constant of the same shape as the operand.
  }];

  // TODO(jpienaar): value's type could be tightened.
  let arguments = (ins AnyAttr:$value, HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;
}

def HLOClient_ErfOp : HLOClient_UnaryElementwiseOp<"erf",
    [NoSideEffect, SameOperandsAndResultShape],
    HLO_FpTensor> {
  let summary = "Erfc operator";

  let description = [{
    Computes the Gauss error function of `x` element-wise.

    erf(x) = erf_impl(x)            if |x| < 1
           = 1 - erfc_impl(x)       otherwise
  }];
}

def HLOClient_ErfcOp : HLOClient_UnaryElementwiseOp<"erfc",
    [NoSideEffect, SameOperandsAndResultShape],
    HLO_FpTensor> {
  let summary = "Erfc operator";

  let description = [{
    Computes an approximation of the error function complement (1 - erf(x)).

    erfc(x) = erfc_impl(x)           if |x| > 1
            = 1 - erf_impl(x)        otherwise
  }];
}

//===----------------------------------------------------------------------===//
// Broadcasting compare op
//===----------------------------------------------------------------------===//

def HLOClient_BroadcastCompareOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_compare", [NoSideEffect]> {
  string summary = "Compare operator (with optional broadcasting)";

  string description = [{
    Compares `lhs` and `rhs` elementwise according to `comparison_direction`
    and `compare_type`. If unspecified, `compare_type` is FLOAT for float element
    types, SIGNED for signed element types and UNSIGNED for unsigned element
    types.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];

  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions,
    HLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<HLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs HLO_PredTensor);

  let builders = [
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs,
      "DenseIntElementsAttr":$broadcast_dimensions,
      "StringAttr":$comparison_direction, CArg<"StringAttr", "{}">:$compare_type)>];
}

#endif  // CHLO_OPS
