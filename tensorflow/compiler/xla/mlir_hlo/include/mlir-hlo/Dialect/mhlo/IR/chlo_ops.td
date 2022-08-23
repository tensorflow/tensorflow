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

#ifndef MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS
#define MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/mhlo/IR/base.td"

def CHLO_Dialect : Dialect {
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

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

include "mlir-hlo/Dialect/mhlo/IR/chlo_ops_enums.td"

class CHLO_Op<string mnemonic, list<Trait> traits> :
    Op<CHLO_Dialect, mnemonic, traits> {
}

class CHLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::chlo::OpTrait";
}

def CHLO_Broadcasting : CHLO_NativeOpTrait<"Broadcasting"> {
}

//===----------------------------------------------------------------------===//
// CHLO binary elementwise op definitions.
// From the client perspective, each of these support both explicit rank
// broadcasting (via the broadcast_dimensions attribute) and implicit degenerate
// shape broadcasting.
//
// These correspond to operations in the chlo and mhlo dialects without the
// "broadcast_" prefix, except that those ops require same-shaped operands and
// results.
//
// See:
//   https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
//   https://www.tensorflow.org/xla/broadcasting
//===----------------------------------------------------------------------===//

class CHLO_BroadcastBinaryElementwiseOp<
  string mnemonic, list<Trait> traits> : CHLO_Op<mnemonic, traits # [
      HLO_BroadcastingElementwise, CHLO_Broadcasting,
      InferTensorTypeWithReify]> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type(results)
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def CHLO_BroadcastAddOp : CHLO_BroadcastBinaryElementwiseOp<"broadcast_add",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Addition operator (with optional broadcasting)";

  string description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastAtan2Op : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_atan2",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Atan2 operator (with optional broadcasting)";

  string description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastDivOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_divide",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Division operator (with optional broadcasting)";

  string description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastMaxOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_maximum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Maximum operator (with optional broadcasting)";

  string description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastMinOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_minimum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Minimum operator (with optional broadcasting)";

  string description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastMulOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_multiply",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Multiplication operator (with optional broadcasting)";

  string description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastNextAfterOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_next_after",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "std::nextafter operator (with optional broadcasting)";

  string description = [{
    Returns the next representable value of `lhs` in the direction of `rhs`,
    element-wise. It can also return a subnormal number.

    Equivalent to the C++ std::nextafter function.
  }];
}

def CHLO_BroadcastPolygammaOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_polygamma", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Polygamma function (with optional broadcasting)";

  let description = [{
    Returns `Polygamma(operand, operand)` element-wise.
  }];
}

def CHLO_BroadcastPowOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_power",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Power operator (with optional broadcasting)";

  string description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastRemOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_remainder",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Remainder operator (with optional broadcasting)";

  string description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastShiftLeftOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_shift_left",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift left operator (with optional broadcasting)";

  string description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastShiftRightArithmeticOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_arithmetic",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right arithmetic operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastShiftRightLogicalOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_logical",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right logical operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastSubOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_subtract",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Subtraction operator (with optional broadcasting)";

  string description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastZetaOp : CHLO_BroadcastBinaryElementwiseOp<
    "broadcast_zeta",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Hurwitz zeta function";

  let description = [{
    Returns `Zeta(operand, operand)` element-wise.

    $$
    \(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\)
    $$
  }];

  let arguments = (ins
    HLO_FpTensor:$lhs,
    HLO_FpTensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
  let results = (outs HLO_FpTensor);
}

//===----------------------------------------------------------------------===//
// XLA binary logical elementwise op definitions.
// The same description as the arithmetic binary elementwise ops applies.
//===----------------------------------------------------------------------===//

class CHLO_BroadcastBinaryLogicalElementwiseOp<string mnemonic> :
    CHLO_BroadcastBinaryElementwiseOp<
      mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def CHLO_BroadcastAndOp: CHLO_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_and"> {
  string summary = "Logical and operator (with optional broadcasting)";

  string description = [{
    Returns `logical_and(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastOrOp: CHLO_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_or"> {
  string summary = "Logical or operator (with optional broadcasting)";

  string description = [{
    Returns `logical_or(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def CHLO_BroadcastXorOp : CHLO_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_xor"> {
  string summary = "Logical xor operator (with optional broadcasting)";

  string description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA non-broadcasting binary operations.
//
// These are operations that are supported by the XLA Builder API but that are
// not part of the HLO compiler instructions as modelled by the MHLO dialect.
//===----------------------------------------------------------------------===//

def CHLO_NextAfterOp : CHLO_Op<"next_after", [NoSideEffect,
    HLO_CompatibleOperandsAndResultType]> {
  let summary = "std::nextafter operator";
  let description = [{
    Returns the next representable value of `x` in the direction of `y`,
    element-wise. It can also return a subnormal number.

    Equivalent to the C++ std::nextafter function.
  }];

  let arguments = (ins HLO_FpTensor:$x, HLO_FpTensor:$y);
  let results = (outs HLO_FpTensor:$result);

  let assemblyFormat = [{
    $x `,` $y attr-dict `:` type($x) `,` type($y) `->` type(results)
  }];
}

def CHLO_PolygammaOp : CHLO_Op<"polygamma", [NoSideEffect,
    HLO_CompatibleOperandsAndResultType]> {
  let summary = "Polygamma function";
  let description = [{
    Returns `Polygamma(operand, operand)` element-wise.
  }];

  let arguments = (ins HLO_FpTensor:$n, HLO_FpTensor:$x);
  let results = (outs HLO_FpTensor:$result);

  let assemblyFormat = [{
    $n `,` $x attr-dict `:` type($n) `,` type($x) `->` type(results)
  }];
}

def CHLO_ZetaOp : CHLO_Op<"zeta", [NoSideEffect,
    HLO_CompatibleOperandsAndResultType]> {
  let summary = "Hurwitz zeta function";
  let description = [{
    Returns `Zeta(operand, operand)` element-wise.

    $$
    \(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\)
    $$
  }];

  let arguments = (ins HLO_FpTensor:$x, HLO_FpTensor:$q);
  let results = (outs HLO_FpTensor:$result);

  let assemblyFormat = [{
    $x `,` $q attr-dict `:` type($x) `,` type($q) `->` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// Broadcasting complex op
//===----------------------------------------------------------------------===//

def CHLO_BroadcastComplexOp : CHLO_BroadcastBinaryElementwiseOp<
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

class CHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type ArgTensorType, Type ResultTensorType> : CHLO_Op<mnemonic,
    traits # [NoSideEffect, Elementwise, SameOperandsAndResultShape,
    InferShapedTypeOpInterface]> {
  let arguments = (ins ArgTensorType:$operand);
  let results = (outs ResultTensorType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
          operands.front(), &reifiedReturnShapes);
    }
  }];
}

def CHLO_AcosOp : CHLO_UnaryElementwiseOp<"acos",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Acos operator";

  let description = [{
    Returns `Acos(operand)` element-wise.

    $$
    \acos(x) = 2 * \atan(\sqrt(1 - x^2) / (1 + x)) if x != -1
             = pi                                  if x == -1
    $$
  }];
}

def CHLO_AcoshOp : CHLO_UnaryElementwiseOp<"acosh",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Acosh operation";

  let description = [{
    Returns `Acosh(operand)` element-wise.

    $$
    \acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
    \acosh(x) = nan                         if x < -1
    $$
  }];
}

def CHLO_AsinOp : CHLO_UnaryElementwiseOp<"asin",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Asin operator";

  let description = [{
    Returns `Asin(operand)` element-wise.

    $$
    \asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
    $$
  }];
}

def CHLO_AsinhOp : CHLO_UnaryElementwiseOp<"asinh",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Asinh operation";

  let description = [{
    Returns `Asinh(operand)` element-wise.

    $$
    \asinh(x) = log(x + sqrt(x^2 + 1))
    $$
  }];
}

def CHLO_AtanOp : CHLO_UnaryElementwiseOp<"atan",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Atan operator";

  let description = [{
    Returns `Atan(operand)` element-wise.

    $$
    \atan(x) = \atan2(x, 1)
    $$
  }];
}

def CHLO_AtanhOp : CHLO_UnaryElementwiseOp<"atanh",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Atanh operator";

  let description = [{
    Returns `Atanh(operand)` element-wise.

    $$
    \atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
              = nan                          otherwise
    $$
  }];
}

def CHLO_BesselI1eOp : CHLO_UnaryElementwiseOp<"bessel_i1e",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Bessel function of order 1";

  let description = [{
    Returns `bessel_i1e(operand)` element-wise.
  }];
}

def CHLO_ConjOp : CHLO_UnaryElementwiseOp<"conj",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Conj operator";

  let description = [{
    Returns `Conj(operand)` element-wise.

    $$
    \conj(x) = (\real(x), \neg(\imag(x)))
    $$
  }];
}

def CHLO_CoshOp : CHLO_UnaryElementwiseOp<"cosh",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Cosh operator";

  let description = [{
    Returns `Cosh(operand)` element-wise.

    $$
    \cosh(x) = (e^x + e^-x) / 2
    $$
  }];
}

def CHLO_SinhOp : CHLO_UnaryElementwiseOp<"sinh",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Sinh operation";

  let description = [{
    Returns `Sinh(operand)` element-wise.

    $$
    \sinh(x) = (e^x - e^-x) / 2                     if |x| < 1
             = e^(x + log(1/2)) - e^(-x + log(1/2)) otherwise.
    $$
  }];
}

def CHLO_TanOp : CHLO_UnaryElementwiseOp<"tan",
    [HLO_CompatibleOperandsAndResultType], HLO_FpOrComplexTensor, HLO_FpOrComplexTensor> {
  let summary = "Tan operation";

  let description = [{
    Returns `Tan(operand)` element-wise.

    $$
    \tan(x) = \sin(x) / \cos(x)
    $$
  }];
}

def CHLO_ConstantOp : CHLO_Op<"constant",
    [ConstantLike, NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs HLO_StaticShapeTensor:$output);

  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def CHLO_ConstantLikeOp : CHLO_Op<"constant_like", [NoSideEffect,
    CHLO_Broadcasting, HLO_BroadcastingElementwise,
    SameOperandsAndResultShape, InferTensorTypeWithReify]> {
  let summary = "Constant like operator";

  let description = [{
    Returns a splat constant of the same shape as the operand.
  }];

  // TODO(jpienaar): value's type could be tightened.
  let arguments = (ins TypedAttrInterface:$value, HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def CHLO_DigammaOp : CHLO_UnaryElementwiseOp<"digamma",
    [HLO_CompatibleOperandsAndResultType], HLO_FpTensor, HLO_FpTensor> {
  let summary = "Digamma function";

  let description = [{
    Returns `Digamma(operand)` element-wise.
  }];
}

def CHLO_ErfOp : CHLO_UnaryElementwiseOp<"erf",
   [HLO_CompatibleOperandsAndResultType], HLO_FpTensor, HLO_FpTensor> {
  let summary = "Erfc operator";

  let description = [{
    Computes the Gauss error function of `x` element-wise.

    erf(x) = erf_impl(x)            if |x| < 1
           = 1 - erfc_impl(x)       otherwise
  }];
}

def CHLO_ErfcOp : CHLO_UnaryElementwiseOp<"erfc",
    [HLO_CompatibleOperandsAndResultType], HLO_FpTensor, HLO_FpTensor> {
  let summary = "Erfc operator";

  let description = [{
    Computes an approximation of the error function complement (1 - erf(x)).

    erfc(x) = erfc_impl(x)           if |x| > 1
            = 1 - erf_impl(x)        otherwise
  }];
}

def CHLO_IsInfOp : CHLO_UnaryElementwiseOp<"is_inf",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_FpTensor,
    HLO_PredTensor> {
  let summary = "IsInf predicate";

  let description = [{
    Returns if a value is +/-inf element-wise.
  }];
}

def CHLO_IsNegInfOp : CHLO_UnaryElementwiseOp<"is_neg_inf",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_FpTensor,
    HLO_PredTensor> {
  let summary = "IsNegInf predicate";

  let description = [{
    Returns if a value is -inf element-wise.
  }];
}

def CHLO_IsPosInfOp : CHLO_UnaryElementwiseOp<"is_pos_inf",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_FpTensor,
    HLO_PredTensor> {
  let summary = "IsPosInf predicate";

  let description = [{
    Returns if a value is +inf element-wise.
  }];
}

def CHLO_LgammaOp : CHLO_UnaryElementwiseOp<"lgamma",
    [HLO_CompatibleOperandsAndResultType], HLO_FpTensor, HLO_FpTensor> {
  let summary = "Lgamma function";

  let description = [{
    Returns `Lgamma(operand)` element-wise.
  }];
}

//===----------------------------------------------------------------------===//
// Broadcasting compare op
//===----------------------------------------------------------------------===//

def CHLO_BroadcastCompareOp : CHLO_BroadcastBinaryElementwiseOp<
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
    CHLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<CHLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs HLO_PredTensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      "DenseIntElementsAttr":$broadcast_dimensions,
      "::mlir::chlo::ComparisonDirection":$comparison_direction,
      CArg<"::mlir::chlo::ComparisonType",
      "::mlir::chlo::ComparisonType::NOTYPE">:$compare_type)>,
  ];
}

//===----------------------------------------------------------------------===//
// Broadcasting select op
//===----------------------------------------------------------------------===//

def CHLO_BroadcastSelectOp : CHLO_Op<"broadcast_select",
    [NoSideEffect, CHLO_Broadcasting, HLO_BroadcastingElementwise,
     InferTensorTypeWithReify]> {
  string summary = "Select operator (with optional numpy-style broadcasting)";

  string description = [{
    Constructs an output array from elements of two input arrays, based on the
    values of a predicate array.

    See https://www.tensorflow.org/xla/operation_semantics#select
  }];

  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_Tensor:$on_true,
    HLO_Tensor:$on_false
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $pred `,` $on_true `,` $on_false attr-dict `:`
    `(` type($pred) `,` type($on_true) `,` type($on_false) `)` `->` type(results)
  }];

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous ops
//===----------------------------------------------------------------------===//

def CHLO_TopKOp : CHLO_Op<"top_k",
    [NoSideEffect, InferTensorType]> {
  string summary = "Finds values and indices of the `k` largest elements for the last dimension";

  string description = [{
If the input is a vector (rank-1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

    values.shape = indices.shape = input.shape[:-1] + [k]

If two elements are equal, the lower-index element appears first.
  }];

  let arguments = (ins
    Arg<HLO_Tensor, [{1-D or higher with last dimension at least `k`.}]>:$operand,
    Arg<I64Attr, [{0-D.  Number of top elements to look for along the last dimension (along each
row for matrices).}]>:$k
  );

  let results = (outs
    HLO_Tensor:$values,
    HLO_Tensor:$indices);

  let assemblyFormat = [{
    `(`$operand `,` `k` `=` $k`)` attr-dict `:`
    type($operand) `->` `(`type($values)`,` type($indices)`)`
  }];
}

//===----------------------------------------------------------------------===//
// Helper ops
//===----------------------------------------------------------------------===//

def CHLO_MinimumBroadcastShapesOp :
    CHLO_Op<"minimum_broadcast_shapes", [NoSideEffect]> {
  string summary = "Minimizes the rank of two or more shapes to be broadcasted";

  string description = [{
    Given two or more 1D tensors representing shapes, returns one 1D tensor for
    each operand, where operand `i` corresponds to output `i`.

    The returned tensors have the property that they specify a shape which is a
    reshape of the corresponding input shape, and the broadcasted output shape
    (using shape::BroadcastOp) of the returned shapes is a reshape of the
    broadcasted output shape of the input shapes. Among all possibilities with
    this property, the one is chosen which minimizes the rank of each returned
    shape.

    The general idea of this op is that it can be used for ops which have a
    broadcasting semantic to operate on shapes with a possibly smaller rank
    while preserving equivalence of the computed values. After computing the
    result of the op using reshaped operands, the result can be reshaped to the
    result that would have been originally computed.

    Here is an example with two input shapes:

    ```mlir
    chlo.minimum_broadcast_shapes [1, 2, 3, 1, 2, 1],
                                     [1, 1, 1, 2, 3] -> [6, 2, 1], [2, 3]
    ```

    The broadcasted output shape of the operands is [1, 2, 3, 1, 2, 3], the
    broadcasted output shape of the outputs is [6, 2, 3]. These two shapes are
    reshapes of each other, and also each output is a reshape of the
    corresponding input.
  }];

  let arguments = (ins Variadic<1DTensorOf<[Index]>>:$shapes);
  let results = (outs Variadic<1DTensorOf<[Index]>>:$results);

  let assemblyFormat = "$shapes attr-dict `:` type($shapes) `->` type($results)";

  let hasVerifier = 1;
}

def CHLO_RankSpecializationClusterOp
    : CHLO_Op<"rank_specialization_cluster", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"RankSpecializationClusterYieldOp">,
    RecursiveSideEffects]> {

  let summary = "Cluster of operations that will be rank-specialized together.";

  let description = [{
    Groups compatible element-wise operatons together so that they can be
    rank-specialized together. The operation takes and yields a variadic number
    of (unranked) tensor operands. Its body region holds one block with one
    block argument per input tensor of the same type. All operations in this
    block must only operate on these block arguments. Results are returned
    through the `rank_specialization_cluster_yield` operation.

    Example:

    ```
    %0 = "chlo.rank_specialization_cluster"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg0_ : tensor<*xf32>, %arg1_ : tensor<*xf32>, %arg2_ : tensor<*xf32>):
      %1 = chlo.broadcast_multiply %arg0_, %arg1_
          : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %2 = chlo.broadcast_add %1, %arg2_
          : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      "chlo.rank_specialization_cluster_yield"(%2) : (tensor<*xf32>) -> ()
    }) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    ```
  }];

  let arguments = (ins Variadic<HLO_Tensor>:$operands);
  let results = (outs Variadic<HLO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
}

def CHLO_RankSpecializationClusterYieldOp
    : CHLO_Op<"rank_specialization_cluster_yield", [NoSideEffect,
    ReturnLike, Terminator, HasParent<"RankSpecializationClusterOp">]> {

  let summary = "Yield operation for `rank_specialization_cluster`";
  let description = [{
    This operation yields the results from within the
    `chlo.rank_specialization_cluster` operation's region. The operation takes
    an arbitrary number of operands and produces no results. The operand number
    and types must match the number and types of the parent
    `rank_specialization_cluster` operation's results.
  }];

  let arguments = (ins Variadic<HLO_Tensor>:$results);
}

def CHLO_DynamicReshapeOp: CHLO_Op<"dynamic_reshape", [NoSideEffect,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface>]> {
  let summary = "Reshape a tensor to a given, possibly dynamic, shape.";
  let description = [{
    Reshapes `operand` to `output_shape`. This allows a single dimension to be
    specified as -1, and it will be computed by the op to create a valid
    reshape.

    Requires:
    - The length of `output_shape` is equal to the rank of `result`.
    - The number of elements in `operand` (that is, the product of extents of
      its shape) is equal to the number of elements in `output_shape` (that is,
      the product of values in `output_shape` with one dimension possibly
      computed).
    - All shape values should be at least -1, and only one extent can be -1.
  }];

  let arguments = (ins HLO_Tensor:$operand, HLO_DimensionTensor:$output_shape);
  let results = (outs HLO_Tensor:$result);
}

#endif  // MLIR_HLO_DIALECT_MHLO_IR_CHLO_OPS
