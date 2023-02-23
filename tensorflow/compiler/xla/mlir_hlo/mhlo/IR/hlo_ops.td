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

// This is the operation definition file for MHLO ops.

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mhlo/IR/hlo_utils.td"
include "mhlo/IR/hlo_ops_common.td"

class MHLO_Op<string mnemonic, list<Trait> traits> :
    Op<MHLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;

  let extraClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

class MHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    MHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

//===----------------------------------------------------------------------===//
// MHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def MHLO_ConstantOp : MHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    MHLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  let hasCustomAssemblyFormat = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def MHLO_IotaOp : MHLO_Op<"iota", [Pure]> {
  let summary = "Iota operator";
  let description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.
  }];
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs MHLO_StaticShapeIntFpOrComplexTensor:$output);

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MHLO_DynamicIotaOp: MHLO_ShapedInterfaceOp<"dynamic_iota", [Pure]> {
  let summary = "Create linear increasing values from 0 to length -1.";
  let description = [{
    Produces an HLO Tensor of the specified shape, with an incremental set of
    values along the specified dimension starting at 0.

    Requires:
    - The output length of the tensor result.
  }];

  let arguments = (ins MHLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs MHLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}


def MHLO_CreateTokenOp : MHLO_Op<"create_token", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create Token operator";

  let description = [{
    Produces a HLO token. Tokens are used for ordering side-effecting operations.
    This is exported to HLO as an AfterAll operation with no operands to
    generate a token.

    Example:

    ```mlir
    %1 = mhlo.create_token : !mhlo.token
    ```
  }];

  let results = (outs MHLO_Token:$output);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// MHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class MHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : MHLO_Op<mnemonic, traits # [Elementwise,
    InferShapedTypeOpInterface, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict
      `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// Abs supports complex to real, so element type is not guaranteed to match.
def MHLO_AbsOp: MHLO_UnaryElementwiseOp<"abs",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
     TensorOf<[HLO_SInt, HLO_Float, HLO_Complex]>,
     TensorOf<[HLO_SInt, HLO_Float]>> {
  let summary = "Absolute value operator";
  let description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.abs %arg0 : tensor<3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_CbrtOp: MHLO_UnaryElementwiseOp<"cbrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Cubic root operator";
  let description = [{
    Returns element-wise cubic root of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.cbrt %arg0 : tensor<3x4xf32>
    ```
  }];
}
def MHLO_CeilOp: MHLO_UnaryElementwiseOp<"ceil",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Ceil operator";
  let description = [{
    Returns `Ceil(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.ceil %arg0 : tensor<2xf32>
    ```
  }];
}
def MHLO_ConvertOp : MHLO_UnaryElementwiseOp<"convert",
    [Pure, SameOperandsAndResultShape], MHLO_Tensor> {
  let summary = "Convert operator";
  let description = [{
    Performs element-wise conversion of values from one type to another, e.g.
    float to int.

    See https://www.tensorflow.org/xla/operation_semantics#convertelementtype.

    Example:

    ```mlir
    %0 = mhlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];

  let hasFolder = 1;

  let hasCanonicalizer = 1;

  let hasCustomHLOConverter = 1;
}

def MHLO_ClzOp: MHLO_UnaryElementwiseOp<"count_leading_zeros",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Count-leading-zeros (Clz) operator";
  let description = [{
    Returns the number of leading zeros in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.count_leading_zeros %arg0 : tensor<2x2xi32>
    ```
  }];
}

def MHLO_CosineOp: MHLO_UnaryElementwiseOp<"cosine",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Cos operator";
  let description = [{
    Returns `Cos(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.cosine %arg0 : tensor<2xf32>
    ```
  }];

  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_ExpOp: MHLO_UnaryElementwiseOp<"exponential",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Exponential operator";
  let description = [{
    Returns `e^(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.exponential %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_Expm1Op: MHLO_UnaryElementwiseOp<"exponential_minus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Exponential minus one operator";
  let description = [{
    Returns `e^(operand) - 1` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.exponential_minus_one %arg0 : tensor<2x2xf32>
    ```
  }];
}
def MHLO_FloorOp: MHLO_UnaryElementwiseOp<"floor",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Floor operator";
  let description = [{
    Returns `Floor(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.floor %arg0 : tensor<2xf32>
    ```
  }];
}
def MHLO_ImagOp: MHLO_UnaryElementwiseOp<"imag",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    MHLO_FpOrComplexTensor, MHLO_FpTensor> {
  let summary = "Imag operator";
  let description = [{
    Returns `Imag(operand)` element-wise.

    Example:

    ```mlir
    %0 = mhlo.imag %arg0 : (tensor<complex<f64>>) -> tensor<f64>
    ```
  }];

  let hasFolder = 1;

}

def MHLO_IsFiniteOp: MHLO_UnaryElementwiseOp<"is_finite", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>], MHLO_Tensor> {
  let summary = "IsFinite operator";
  let description = [{
    Tests whether each element of operand is finite, i.e., is not positive or
    negative infinity, and is not NaN. Returns a tensor of 1-bit integers with
    the same shape as the input, where each element is nonzero (i.e. true) if
    and only if the corresponding input element is finite.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.is_finite %arg0 : (tensor<2xf32>) -> tensor<2xi1>
    ```
  }];
  let arguments = (ins MHLO_FpTensor:$x);
  let results = (outs MHLO_PredTensor:$y);

  let assemblyFormat = [{
    $x attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_LogOp: MHLO_UnaryElementwiseOp<"log",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Logarithm operator";
  let description = [{
    Returns `log(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.log %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_Log1pOp: MHLO_UnaryElementwiseOp<"log_plus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Log1p operator";
  let description = [{
    Returns `log(operand+1)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.log_plus_one %arg0 : tensor<2xf32>
    ```
  }];
}
def MHLO_LogisticOp: MHLO_UnaryElementwiseOp<"logistic",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Logistic operator";
  let description = [{
    Returns `logistic(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.logistic %arg0 : tensor<2x2xf32>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_NotOp: MHLO_UnaryElementwiseOp<"not",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_PredOrIntTensor> {
  let summary = "Not operator";
  let description = [{
    Returns biwise-NOT of `operand` element-wise. The input tensor must be
    of type integer `MHLO_Int` or boolean `MHLO_Pred`.

    Note: For boolean tensor, the bitwise-NOT is equivalent to logical-NOT.

    Example:

    ```mlir
    %0 = mhlo.not %arg0 : tensor<5x3x1xi1>
    ```
  }];
    let hasFolder = 1;
}

def MHLO_NegOp: MHLO_UnaryElementwiseOp<"negate",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexTensor> {
  let summary = "Negation operator";
  let description = [{
    Returns `-operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.negate %arg0 : tensor<2x3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PopulationCountOp: MHLO_UnaryElementwiseOp<"popcnt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "PopulationCount operator";
  let description = [{
    Returns the number of bits set in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.popcnt %arg0 : tensor<2x2xi32>
    ```
  }];
}
def MHLO_RealOp: MHLO_UnaryElementwiseOp<"real",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    MHLO_FpOrComplexTensor, MHLO_FpTensor> {
  let summary = "Real operator";
  let description = [{
    Returns `Real(operand)` element-wise.

    Example:

    ```mlir
    %1 = mhlo.real %arg2 : (tensor<complex<f32>>) -> tensor<f32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RoundOp: MHLO_UnaryElementwiseOp<"round_nearest_afz",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Round operator, ties away from zero";
  let description = [{
    Returns `Round(operand)` element-wise, rounding to nearest integer with
    half-way cases rounding away from zero.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.round_nearest_afz %11 : tensor<10x10xbf16>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RoundNearestEvenOp: MHLO_UnaryElementwiseOp<"round_nearest_even",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Round operator, ties to even";
  let description = [{
    Returns `Round(operand)` element-wise, rounding to nearest integer with
    half-way cases rounding towards even numbers.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.round_nearest_even %arg0 : tensor<10x10xbf16>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RsqrtOp: MHLO_UnaryElementwiseOp<"rsqrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Reciprocal Square-root operator";
  let description = [{
    Returns `1.0 / sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.rsqrt %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_SignOp: MHLO_UnaryElementwiseOp<"sign",
    [Pure, HLO_CompatibleOperandsAndResultType],
    TensorOf<[MHLO_SInt, MHLO_Float, MHLO_Complex]>> {
  let summary = "Sign operator";
  let description = [{
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

    Example:

    ```mlir
    %0 = mhlo.sign %arg0 : tensor<1x2x3x4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_SineOp: MHLO_UnaryElementwiseOp<"sine",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Sin operator";
  let description = [{
    Returns `Sin(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.sine %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_TanOp: MHLO_UnaryElementwiseOp<"tan",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Tan operator";
  let description = [{
    Returns `Tan(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.tan %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_SqrtOp: MHLO_UnaryElementwiseOp<"sqrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Square-root operator";
  let description = [{
    Returns `sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.sqrt %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_TanhOp: MHLO_UnaryElementwiseOp<"tanh",
    [Pure, HLO_CompatibleOperandsAndResultType],
    MHLO_FpOrComplexTensor> {
  let summary = "Tanh operator";
  let description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.

    Example:

    ```mlir
    %0 = mhlo.tanh %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}
//===----------------------------------------------------------------------===//
// MHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class MHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType = MHLO_Tensor, Type ResultType = OperandType> :
    MHLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def MHLO_AddOp : MHLO_BinaryElementwiseOp<"add",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Addition operator";
  let description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = mhlo.add %arg2, %arg3 : (tensor<4xf32, #CSR>, tensor<4xf32, #DCSR>) -> tensor<4xf32, #CSR>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_Atan2Op : MHLO_BinaryElementwiseOp<"atan2",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Atan2 operator";
  let description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.atan2 %arg0, %arg1 : tensor<4xi32>
    ```
  }];
}

def MHLO_ComplexOp: MHLO_BinaryElementwiseOp<"complex", [Pure,
    SameOperandsElementType, SameOperandsAndResultShape,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Complex operator";
  let description = [{
    Performs element-wise conversion of a pair of real and imaginary values to
    a complex value.

    Example:

    ```mlir
    %0 = mhlo.complex %arg0, %arg0 : tensor<4xcomplex<f32>>
    ```
  }];

  let arguments = (ins MHLO_Fp32Or64Tensor:$lhs, MHLO_Fp32Or64Tensor:$rhs);
  let results = (outs MHLO_ComplexTensor:$result);

  let hasFolder = 1;

  let assemblyFormat = [{
    operands attr-dict
      `:` custom<ComplexOpType>(type($lhs), type($rhs), type($result))
  }];
}

def MHLO_DivOp : MHLO_BinaryElementwiseOp<"divide",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexTensor> {
  let summary = "Division operator";
  let description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.divide %arg0, %arg0 : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MaxOp : MHLO_BinaryElementwiseOp<"maximum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Maximum operator";
  let description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.maximum %arg0, %arg1 : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MinOp : MHLO_BinaryElementwiseOp<"minimum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Minimum operator";
  let description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.minimum %arg0, %arg1 : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MulOp : MHLO_BinaryElementwiseOp<"multiply",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Multiplication operator";
  let description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.multiply %arg0, %arg0 : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PowOp : MHLO_BinaryElementwiseOp<"power",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexTensor> {
  let summary = "Power operator";
  let description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.power %arg0, %arg0 : tensor<2xf32>
    ```
  }];
}
def MHLO_RemOp : MHLO_BinaryElementwiseOp<"remainder",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexTensor> {
  let summary = "Remainder operator";
  let description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.remainder %arg0, %arg1 : tensor<4xi64>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_ShiftLeftOp : MHLO_BinaryElementwiseOp<"shift_left",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Shift Left operator";
  let description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.shift_left %arg0, %arg1 : tensor<4xi32>
    ```
  }];
}

def MHLO_ShiftRightArithmeticOp : MHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Shift right arithmetic operator";
  let description = [{
    Returns arithmetic `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
    ```

  }];
}

def MHLO_ShiftRightLogicalOp : MHLO_BinaryElementwiseOp<"shift_right_logical",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Shift right logical operator";
  let description = [{
    Returns logical `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>
    ```
  }];
}

def MHLO_SubtractOp : MHLO_BinaryElementwiseOp<"subtract",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexTensor> {
  let summary = "Subtraction operator";
  let description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.

    Example:

    ```mlir
    %0 = mhlo.subtract %arg0, %arg0 : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

// TODO(b/232442915): Implement stochastic_convert MHLO once HLO interface is submitted.
def MHLO_StochasticConvertOp : MHLO_Op<"stochastic_convert",
      [Pure, AllShapesMatch<["operand", "random", "result"]>]> {
  let summary = "Stochastic convert operator";
  let description = [{
    Performs element-wise conversion of values from one bigger type to a smaller
    one with stochastic rounding using the random number passed in.
  }];

  let arguments = (ins MHLO_FpTensor:$operand, TensorOf<[MHLO_UInt]>:$random);
  let results = (outs MHLO_Tensor:$result);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MHLO binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class MHLO_BinaryBiwiseOrLogicalElementwiseOp<string mnemonic> :
        MHLO_BinaryElementwiseOp<mnemonic,
          [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let arguments = (ins
    MHLO_PredOrIntTensor:$lhs,
    MHLO_PredOrIntTensor:$rhs
  );

  let hasFolder = 1;
}

def MHLO_AndOp: MHLO_BinaryBiwiseOrLogicalElementwiseOp<"and"> {
  let summary = "And operator";
  let description = [{
    Returns biwise-AND of `lhs` and `rhs` element-wise. The input tensors must
    be of type integer `MHLO_Int` or boolean `MHLO_Pred`.

    Note: For boolean tensor, the bitwise-AND is equivalent to logical-AND.

    Example:

    ```mlir
    %0 = mhlo.and %arg0, %arg1 : tensor<i1>
    ```
  }];
}

def MHLO_OrOp: MHLO_BinaryBiwiseOrLogicalElementwiseOp<"or"> {
  let summary = "Or operator";
  let description = [{
    Returns biwise-OR of `lhs` and `rhs` element-wise. The input tensors must
    be of type integer `MHLO_Int` or boolean `MHLO_Pred`.

    Note: For boolean tensor, the bitwise-OR is equivalent to logical-OR.

    Example:

    ```mlir
    %0 = mhlo.or %arg0, %arg1 : tensor<2xi1>
    ```
  }];
}

def MHLO_XorOp : MHLO_BinaryBiwiseOrLogicalElementwiseOp<"xor"> {
  let summary = "Xor operator";
  let description = [{
    Returns biwise-XOR of `lhs` and `rhs` element-wise. The input tensors must
    be of type integer `MHLO_Int` or boolean `MHLO_Pred`.

    Note: For boolean tensor, the bitwise-XOR is equivalent to logical-XOR.

    Example:

    ```mlir
    %0 = mhlo.xor %arg0, %arg1 : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MHLO communication op definitions.
//===----------------------------------------------------------------------===//

// InfeedOp corresponds to 'InfeedWithToken' xla client API and not 'Infeed'.
// InfeedWithToken allows ordering of infeed HLO instructions using tokens.
def MHLO_InfeedOp : MHLO_Op<"infeed", []> {

  let summary = "Infeed operator";

  let description = [{
    Reads a single data item from the implicit Infeed streaming interface of
    the device, interpreting the data as the given shape, and returns a XlaOp
    of the data. Multiple Infeed operations are allowed in a computation, but
    there must be a total order among the Infeed operations.

    Attributes:
      layout:  Array attribute. Each element of the array is a minor_to_major
               array corresponding to the shape of the data read from the infeed
               interface.

    See https://www.tensorflow.org/xla/operation_semantics#infeed.
  }];

  let arguments = (ins
    MHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<MHLO_StaticShapeTensorOrToken>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

// OutfeedOp corresponds to 'OutfeedWithToken' xla client API and not 'Outfeed'.
// OutfeedWithToken allows ordering of outfeed HLO instructions using tokens.
def MHLO_OutfeedOp : MHLO_Op<"outfeed",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Outfeed operator";

  let description = [{
    Generates outgoing data transfers for the given data. It takes data and a
    token type operand and produces a token type value. Tokens are used for
    ordering side-effecting operations.

    See https://www.tensorflow.org/xla/operation_semantics#outfeed.
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    MHLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs MHLO_Token);
  let hasCustomHLOConverter = 1;
}

def MHLO_SendOp : MHLO_Op<"send",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "Send operator";

  let description = [{
    Sends the given operand data to a Recv instruction in another computation
    that shares the same channel handle. Does not return any data. Similar to
    the Recv operation, Send operation represents synchronous communication,
    and is internally decomposed into 2 HLO instructions (Send and SendDone) to
    enable asynchronous data transfers.

    See https://www.tensorflow.org/xla/operation_semantics#send.
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    MHLO_Token:$token,
    MHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs MHLO_Token);
  let hasCustomHLOConverter = 1;
}

def MHLO_RecvOp : MHLO_Op<"recv", []> {

  let summary = "Recv operator";

  let description = [{
    Receives data of the given shape from a Send instruction in another
    computation that shares the same channel handle. Returns a tuple containing
    value for the received data and a token. Recv operation represents
    synchronous communication. However, the instruction is internally decomposed
    into 2 HLO instructions (Recv and RecvDone) to enable asynchronous data
    transfers.

    See https://www.tensorflow.org/xla/operation_semantics#recv.
  }];

  let arguments = (ins
    MHLO_Token:$token,
    MHLO_ChannelHandle:$channel_handle,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs Variadic<MHLO_StaticShapeTensorOrToken>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MHLO parallelism related op definitions.
//===----------------------------------------------------------------------===//

def MHLO_ReplicaIdOp : MHLO_Op<"replica_id", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ReplicaId operator";
  let description = [{
    Returns the unique ID (int32 scalar) of the replica.

    The unique ID of each replica is an unsigned integer in the interval [0, N),
    where N is the number of replicas. Since all the replicas are running the
    same program, a ReplicaId() call in the program will return a different
    value on each replica.

    See https://www.tensorflow.org/xla/operation_semantics#replicaid.

    Example:

    ```mlir
    %0 = mhlo.replica_id : tensor<ui32>
    ```
  }];
  let results = (outs TensorOf<[UI32]>);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// MHLO control flow op definitions.
//===----------------------------------------------------------------------===//

def MHLO_AddDependencyOp : MHLO_Op<"add_dependency", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AddDependency operator";
  let description = [{
    AddDependency takes two operands: a data operand and a token. The output of
    the operation is the data operand. When used with AfterAll this operation
    enables ordering non-side-effecting operations (those that do not produce
    token values).

    Example:

    ```mlir
    %1 = mhlo.add_dependency %arg0, %0 : (tensor<3x4xf32>, !mhlo.token) -> tensor<3x4xf32>
    ```
  }];

  let arguments = (ins MHLO_TensorOrToken:$operand, MHLO_Token:$token);
  let results = (outs MHLO_TensorOrToken:$output);
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}


def MHLO_AfterAllOp : MHLO_Op<"after_all", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

  let summary = "AfterAll operator";

  let description = [{
    AfterAll takes a variadic number of tokens and produces a single token.
    Tokens are primitive types which can be threaded between side-effecting
    operations to enforce ordering. AfterAll can be used as a join of tokens
    for ordering a operation after a set operations.

    See https://www.tensorflow.org/xla/operation_semantics#afterall.

    Example:

    ```mlir
    %0 = mhlo.after_all %arg0, %arg1 : !mhlo.token
    ```
  }];

  let arguments = (ins Variadic<MHLO_Token>:$inputs);
  let results = (outs MHLO_Token:$result);

  let assemblyFormat = [{
    $inputs attr-dict
      `:` custom<VariadicSameOperandsAndResultType>(ref($inputs), type($inputs), type($result))
  }];
}

def MHLO_AsyncStartOp : MHLO_Op<"async_start", []> {
  let summary = "AsyncStart operator";

  let description = [{
    AsyncStart kicks off an asynchronous computation.

    This is used when there are functions that contain both asynchronous waits
    (such as DMAs) and on-thread computation. For example, a function might
    consist of a computation, a DMA, another computation, a second DMA, and a
    final computation. This would be represented as an async_start followed by
    and async_update and an async_done. The async_start would do the first
    computation on-thread and then start the DMA. The async_update would wait
    for the DMA to complete if it wasn't yet done, then execute the second
    computation in the function, and start the second DMA. Finally, the
    async_done would wait on this last DMA, and then run the last computation
    that needs to be run on-thread and return the result of that final
    computation.

    `operands` are passed to the computation directly
    `called_computation` is the function that will be run asynchronously
    `execution_thread` is the name of the thread in which it will be run. The main
      thread is called "main". All threads have names.
    `group_id` labels a set of async-start, async-done, and zero or more
      async-update ops corresponding to the same computation. We
      represent a missing group_id with either an negative value or None.

    This returns all the state needed between async ops. After buffer
    assignment, the return values represents the space needed to hold the input,
    results, and any scratchpads needed or edited by the async op.
  }];

  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple>:$inputs,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread,
    OptionalAttr<I64Attr>:$group_id
  );

  let results = (outs MHLO_AsyncBundle);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_AsyncUpdateOp : MHLO_Op<"async_update", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AsyncUpdate operator";

  let description = [{
    AsyncUpdate blocks on an asynchronous computation until a sync barrier.
    This returns `bundle` after operating on it.

    See the documentation for AsyncStart for more information.
  }];

  let arguments = (ins
    MHLO_AsyncBundle:$bundle,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread,
    OptionalAttr<I64Attr>:$group_id
    );

  let results = (outs MHLO_AsyncBundle);

  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_AsyncDoneOp : MHLO_Op<"async_done", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AsyncDone operator";

  let description = [{
    AsyncDone blocks until the end of an asynchronous computation.
    It returns the final result of the asynchronous computation.

    See the documentation for AsyncStart for more information.
  }];

  let arguments = (ins
    MHLO_AsyncBundle:$bundle,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread,
    OptionalAttr<I64Attr>:$group_id
    );

  let results = (outs Variadic<MHLO_TensorOrTokenOrTuple>);
  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. IfOp maps to predicated
// conditional use of kConditional HLO.
def MHLO_IfOp: MHLO_Op<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "If operator";

  let description = [{
    Executes the function `true_branch` if `pred` is true or `false_branch` if
    pred is false, and returns the result.

    The type of the returned values of `true_branch` and `false_branch`
    functions must be the same and equal to the types of the values returned by
    the operation.

    Note that only one of two functions will be executed depending on the value
    of `pred`.
  }];

  let arguments = (ins
    MHLO_PredTensor:$pred
  );

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  // TODO(b/129422361): ConditionalOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def MHLO_CaseOp: MHLO_Op<"case", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Switch-Case operator";
  let description = [{
    Returns the result of executing `branches[index]`. If `index` is < 0 or >=
    N, then `branches[N-1]` is executed as the default branch.

    The type of the returned values of each branch must be the same and equal
    to the types of the values returned by the operation.

    Note that only one of the branches will be executed depending on the value
    of index.
  }];

  let arguments = (ins
    I32Tensor:$index
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
}

def MHLO_WhileOp: MHLO_Op<"while", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>,
      OpAsmOpInterface
    ]> {
  let summary = "While operator";
  let description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];
  let arguments = (ins Variadic<MHLO_TensorOrToken>:$operand);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let results = (outs Variadic<MHLO_TensorOrToken>);

  let extraClassDeclaration = [{
    // Method of OpAsmOpInterface used during custom printing to name the block
    // arguments in the nested regions. We name both the condition and the body
    // regions entry arguments the same way, with a `iterArg` prefix. Since the
    // two regions are side-by-side they will have the same name, which allows
    // us to print them once and share it for the two regions, and still be able
    // to parse them back.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn setNameFn) {
      for (BlockArgument arg : region.getArguments())
        setNameFn(arg, "iterArg");
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MHLO_AllGatherOp : MHLO_Op<"all_gather", [SameOperandsAndResultElementType]> {

  string summary = "AllGather operator";

  string description = [{
    Performs concatenation across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allgather
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs MHLO_Tensor);
  // use_global_device_ids is rarely used, so we add simplified builder methods
  // for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::IntegerAttr":$all_gather_dim,
      "::mlir::DenseIntElementsAttr":$replica_groups,
      "::mlir::mhlo::ChannelHandleAttr":$channel_handle)>];

  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_AllReduceOp : MHLO_Op<"all_reduce",
    [HLO_CompatibleOperandsAndResultType]> {
  let summary = "AllReduce operator";
  let description = [{
    Performs a custom reduction across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allreduce.
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs MHLO_Tensor);
  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_ReduceScatterOp : MHLO_Op<"reduce_scatter",
    [SameOperandsAndResultElementType]> {
  let summary = "ReduceScatter operator";
  let description = [{
     Performs all_reduce followed by a scatter.

     See https://www.tensorflow.org/xla/operation_semantics#reducescatter
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64Attr:$scatter_dimension,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs MHLO_Tensor);
  // use_global_device_ids is rarely used, so we add simplified builder methods
  // for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::IntegerAttr":$scatter_dimension,
      "::mlir::DenseIntElementsAttr":$replica_groups,
      "::mlir::mhlo::ChannelHandleAttr":$channel_handle)>];

  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_AllToAllOp : MHLO_Op<"all_to_all",
    [Pure, SameOperandsElementType, SameOperandsShape, SameVariadicOperandSize,
     InferTensorType]> {

  let arguments = (ins
    // ArrayAllToAll must have exactly one operand, TupleAllToAll at least one.
    Variadic<MHLO_Tensor>:$operand,
    // split_dimension, concat_dimension and split_count are present for array
    // all-to-all, absent for tuple all-to-all.
    OptionalAttr<I64Attr>:$split_dimension,
    OptionalAttr<I64Attr>:$concat_dimension,
    OptionalAttr<I64Attr>:$split_count,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs Variadic<MHLO_Tensor>);
  let hasCustomHLOConverter = 1;

  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::IntegerAttr": $split_dimension,
      "::mlir::IntegerAttr": $concat_dimension,
      "::mlir::IntegerAttr": $split_count,
      "::mlir::DenseIntElementsAttr": $replica_groups)>,
    OpBuilder<(ins
      "::mlir::TypeRange":$result_type, "::mlir::ValueRange":$operand,
      "::mlir::IntegerAttr": $split_dimension,
      "::mlir::IntegerAttr": $concat_dimension,
      "::mlir::IntegerAttr": $split_count,
      "::mlir::DenseIntElementsAttr": $replica_groups)>];
}

def MHLO_ReduceOp: MHLO_ShapedInterfaceOp<"reduce", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType,
    ]> {
  let summary = "Reduce operator";
  let description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    Variadic<MHLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<MHLO_Tensor>);

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def MHLO_GetTupleElementOp: MHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "GetTupleElement operator";
  let description = [{
    Returns a member of a tuple specified by an index.

    See https://www.tensorflow.org/xla/operation_semantics#gettupleelement.
  }];
  let arguments = (ins
    MHLO_Tuple:$operand,
    I32Attr:$index
  );

  let results = (outs MHLO_TensorOrTokenOrTuple);

  let hasFolder = 1;

  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_TupleOp : MHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "XLA's tuple op";
  let description = [{
     Groups a set of tensor inputs into a single tuple object.

     See https://www.tensorflow.org/xla/operation_semantics#tuple.

    Example:

    ```mlir
    %0 = mhlo.tuple %arg0, %arg0 : tuple<tensor<i32>, tensor<i32>>
    ```
   }];
  let arguments = (ins Variadic<MHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs MHLO_Tuple:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

def MHLO_CompareOp: MHLO_Op<"compare", [Pure, SameOperandsElementType,
    SameOperandsAndResultShape, Elementwise, InferTensorTypeWithReify]> {
  let summary = "Comparison operator";
  let description = [{
    Compares `lhs` and `rhs` elementwise according to `comparison_direction`
    and `compare_type`. If unspecified, `compare_type` is FLOAT for float element
    types, SIGNED for signed element types and UNSIGNED for unsigned element
    types.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.

    Example:

    ```mlir
    %0 = mhlo.compare LT, %arg0, %arg1 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %1 = mhlo.compare LT, %arg0, %arg1, TOTALORDER : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<MHLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs MHLO_PredTensor);

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      "::mlir::mhlo::ComparisonDirection":$comparison_direction,
      CArg<"::mlir::mhlo::ComparisonType",
      "::mlir::mhlo::ComparisonType::NOTYPE">:$compare_type)>,
  ];

  let hasCustomHLOConverter = 1;

  let assemblyFormat = [{
    $comparison_direction `,` $lhs `,` $rhs (`,` $compare_type^)?
      attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MHLO Slice definitions.
//===----------------------------------------------------------------------===//

def MHLO_SliceOp: MHLO_Op<
      "slice",
      [Pure, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_DynamicSliceOp: MHLO_Op<"dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]>,
       InferTensorType]> {
  let summary = "Dynamic Slice operator";
  let description = [{
    Extracts a sub-array from the input array at dynamic start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicslice.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    Variadic<MHLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs MHLO_Tensor:$result);
  let hasCanonicalizer = 1;
}

def MHLO_DynamicUpdateSliceOp: MHLO_Op<"dynamic_update_slice",
      [Pure, AllElementTypesMatch<["operand", "update", "result"]>,
       InferTensorType]> {
  let summary = "Dynamic Update Slice operator";
  let description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.

    Example:

    ```mlir
    %0 = mhlo.dynamic_update_slice %arg0, %arg1, %arg2
           : (tensor<4xf32>, tensor<2xf32>, tensor<i32>) -> tensor<4xf32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$update,
    Variadic<MHLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs MHLO_Tensor:$result);
  let hasFolder = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}


//===----------------------------------------------------------------------===//
// MHLO Other op definitions.
//===----------------------------------------------------------------------===//

def MHLO_DomainOp : MHLO_Op<"domain", [HLO_CompatibleOperandsAndResultType, InferTypeOpInterface, Pure]> {
  let summary = "Marks groups of instructions (domains) with a property";
  let description = [{
    Domain instructions are used to group instructions with the same
    DomainMetadata property. ShardingMetadata is the main use case today to
    group instructions on the same device. Domain instructions provide two
    major benefits:
      - Prevent unintentionally optimizing instructions across domains.
      - Automatically assign the metadata of the instructions created in the domain.
    Without domain instructions, each HLO optimization pass would have to check
    and propagate the metadata, which would be easy to miss and also adds
    complexity to the compiler. Since domain instructions connect two different
    domains, each domain instruction is associated with two DomainMetadata --
    one on the operand side and one on the user side of the domain.
  }];
  let arguments = (ins
    MHLO_TensorOrToken:$operand,
    MHLO_DomainKindAttr:$kind,
    StrAttr:$entry_metadata,
    StrAttr:$exit_metadata
  );
  let results = (outs MHLO_TensorOrToken:$result);
  let hasCustomHLOConverter = 1;
}

def MHLO_BatchNormGradOp : MHLO_Op<"batch_norm_grad", [Pure,
    AllElementTypesMatch<["operand", "grad_operand", "grad_scale", "grad_offset"]>,
    InferTensorType]> {
  let summary = "Batch Normalization Gradient";
  let description = [{
    Calculates gradients of batch norm.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
  }];

  let arguments = (ins
    RankedTensorOf<[MHLO_Float]>:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$mean,
    1DTensorOf<[MHLO_Float]>:$variance,
    RankedTensorOf<[MHLO_Float]>:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      RankedTensorOf<[MHLO_Float]>:$grad_operand,
      1DTensorOf<[MHLO_Float]>:$grad_scale,
      1DTensorOf<[MHLO_Float]>:$grad_offset);

  let hasCustomHLOConverter = 1;
}

def MHLO_BatchNormInferenceOp : MHLO_Op<"batch_norm_inference",
    [Pure, AllElementTypesMatch<["operand", "result"]>, InferTensorType]> {
  let summary = "Batch Normalization for Inference";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];

  let arguments = (ins
    RankedTensorOf<[MHLO_Float]>:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$offset,
    1DTensorOf<[MHLO_Float]>:$mean,
    1DTensorOf<[MHLO_Float]>:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs RankedTensorOf<[MHLO_Float]>:$result);
}

def MHLO_BatchNormTrainingOp : MHLO_Op<"batch_norm_training",
    [Pure, AllElementTypesMatch<["operand", "output", "batch_mean", "batch_var"]>,
    InferTensorType]> {
  let summary = "Batch Normalization for Training";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
  }];

  let arguments = (ins
    RankedTensorOf<[MHLO_Float]>:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      RankedTensorOf<[MHLO_Float]>:$output,
      1DTensorOf<[MHLO_Float]>:$batch_mean,
      1DTensorOf<[MHLO_Float]>:$batch_var);

  let hasCustomHLOConverter = 1;
}

def MHLO_BitcastConvertOp : MHLO_ShapedInterfaceOp<"bitcast_convert",
    [Pure]> {
  let summary = "BitcastConvert operator";
  let description = [{
    Similar to a 'tf.bitcast' in TensorFlow, performs an element-wise bitcast
    operation from a data shape to a target shape. The dimensions must match,
    and the conversion is an element-wise one. Bitcast is implemented as a
    low-level cast, so machines with different floating-point representations
    will give different results.

    See https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype.

    Example:

    ```mlir
    %0 = mhlo.bitcast_convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand);
  let results = (outs MHLO_Tensor);
  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_BroadcastOp : MHLO_ShapedInterfaceOp<"broadcast",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Broadcast a tensor to a higher rank by prepending dimensions";
  let description = [{
    Broadcasts the operand tensor to a higher rank by prepending
    `broadcast_sizes` to the dimensions. The current values of the operand are
    copied into the other dimensions.

    This is a more limited form of broadcasting, that corresponds to the XLA
    client Broadcast method. For a more general form of broadcasting, see the
    BroadcastInDimOp.

    See https://www.tensorflow.org/xla/operation_semantics#broadcast.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_BroadcastInDimOp : MHLO_Op<"broadcast_in_dim",
      [Pure, SameOperandsAndResultElementType]> {
  let summary = "Broadcast a tensor into the given shape by adding dimensions.";
  let description = [{
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
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs MHLO_StaticShapeTensor);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  // Only handles a static subset of the legacy format.
  let hasCustomHLOConverter = 1;
}

def MHLO_DynamicBroadcastInDimOp : MHLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim", [Pure]> {
  let summary = "Broadcast a tensor into the given dynamic shape by adding dimensions.";
  let description = [{
    This is a generalization of the BroadcastInDimOp which accepts its output
    dimensions as an argument. It should eventually supercede the statically
    shaped original, but is being phased as a separate op in order to support
    compatibility with lowerings and translations that precede dynamic shapes.

    The op accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_DimensionTensor:$output_dimensions,
    MHLO_BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<MHLO_BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<MHLO_BroadcastDimAttr>:$known_nonexpanding_dimensions
  );

  let results = (outs MHLO_Tensor);

  let builders = [
    OpBuilder<(ins
        "Type":$result_type, "Value":$operand, "Value":$output_dimensions,
        "DenseIntElementsAttr":$broadcast_dimensions), [{
      build($_builder, $_state, result_type, operand, output_dimensions,
          broadcast_dimensions, /*known_expanding_dimensions=*/{},
          /*known_nonexpanding_dimensions=*/{});
    }]>
  ];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}

// Note: There is no MHLO_CallOp because the standard call operation mlir::func::CallOp
// is used instead. A mlir::func::CallOp is exported to a HLO call instruction
// directly.

def MHLO_CholeskyOp : MHLO_Op<"cholesky",
      [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Cholesky operator";
  let description = [{
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
  let arguments = (ins
    MHLO_FpOrComplexTensor:$a,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$lower
  );

  let results = (outs MHLO_FpOrComplexTensor);
}

def MHLO_ClampOp : MHLO_ShapedInterfaceOp<"clamp", [Pure,
  SameOperandsAndResultElementType, HLO_BroadcastingElementwise,
  InferTensorType]> {
  let summary = "Clamp operator";
  let description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.

    Example:

    ```mlir
    %0 = mhlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %1 = mhlo.clamp %arg1, %arg1, %arg1 : tensor<4xf32>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$min,
    MHLO_Tensor:$operand,
    MHLO_Tensor:$max
  );
  let results = (outs MHLO_Tensor:$result);

  let hasFolder = 1;

  let assemblyFormat = [{
    $min `,` $operand `,` $max attr-dict
      `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
  }];
}

def MHLO_ConcatenateOp : MHLO_ShapedInterfaceOp<"concatenate",
    [Pure, SameOperandsAndResultElementType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "XLA's concatenate op";
  let description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$val,
    I64Attr:$dimension
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_CollectivePermuteOp: MHLO_Op<"collective_permute",
    [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "CollectivePermute operator";
  let description = [{
    CollectivePermute is a collective operation that sends and receives data
    cross replicas.
    Note that there are the following restrictions on the source_target_pair:
    - Any two pairs should not have the same target replica id, and they should
    not have the same source replica id.
    - If a replica id is not a target in any pair, then the output on that
    replica is a tensor consists of 0(s) with the same shape as the input.

    See https://www.tensorflow.org/xla/operation_semantics#collectivepermute.

  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs MHLO_Tensor);
  let hasVerifier = 1;
  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$source_target_pairs)>];
}

def MHLO_ConvolutionOp : MHLO_Op<"convolution", [Pure]> {
  let summary = "Convolution operator";
  let description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];
  let arguments = !con(
    (ins
       MHLO_Tensor:$lhs,
       MHLO_Tensor:$rhs),
    MHLO_ConvolutionAttributes.attributes);

  let results = (outs MHLO_Tensor);
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  code extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = getWindowReversalAttr();
      return reversal && llvm::any_of(reversal.getValues<bool>(),
                                      [](bool v) { return v; });
    }
  }];

 let assemblyFormat = [{
    `(`operands`)`
       `dim_numbers` `=` custom<ConvolutionDimensions>($dimension_numbers) `,`
       `window` `=` `{` custom<WindowAttributes>($window_strides, $padding,
                                                 $lhs_dilation, $rhs_dilation,
                                                 $window_reversal) `}`
       attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_CopyOp: MHLO_Op<"copy", [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Copy operator";
  let description = [{
    Returns a copy of `operand`.

    Example:

    ```mlir
    %0 = mhlo.copy %arg0 : tensor<f32>
    ```
  }];
  let arguments = (ins
      MHLO_TensorOrTokenOrTuple:$operand,
      OptionalAttr<I32Attr>:$cross_program_prefetch_index
  );
  let results = (outs MHLO_TensorOrTokenOrTuple:$result);
  let hasCustomHLOConverter = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
    operands attr-dict
      `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def MHLO_CrossReplicaSumOp : MHLO_Op<"cross-replica-sum",
    [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Sums input across replicated instances.";
  let description = [{
     For each of the replica groups, operands of the group devices are summed
     so that each device has the sum.

     For example, suppose there are 8 TPU devices: `[A, B, C, D, E, F, G, H]`.
     Passing group_assignment=`[[0,2,4,6],[1,3,5,7]]` sets `A, C, E, G` as group 0,
     and `B, D, F, H` as group 1. Thus we get the outputs:
     `[A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H, A+C+E+G, B+D+F+H]`.

     See https://www.tensorflow.org/xla/operation_semantics#crossreplicasum.
   }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs MHLO_Tensor);
}

def MHLO_CustomCallOp: MHLO_Op<"custom_call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "CustomCall operator";
  let description = [{
    A custom call invokes code external to XLA. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    If XLA runtime is enabled for a backend, then custom calls use the runtime
    custom call calling convention to call into the external functions. This
    calling convention defines an ABI for encoding arguments, attributes and
    results.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.

    Depending on the API version there are two ways to pass extra bits of static
    information to the external function:

    1. For `API_VERSION_TYPED_FFI` custom calls `backend_config` must be a
       dictionary attribute, that will be encoded according to the custom call
       calling convention and passed to the external function as the attributes
       argument. External code is expected to use declarative bindings (see
       `xla/runtime/custom_call.h`) to decode them at run time.

    2. For previous API versions it is the user responsibility to encode extra
       bits of static information as a string `backend_config` attribute, and
       decode it at run time.

    `API_VERSION_TYPED_FFI` custom calls only supported if XLA uses XLA runtime.

    `backend_config` can encode arbitrarily large amounts of information.

    `has_side_effect` must be true if the custom call has side-effects.
    `api_version` specifies the version of the API used by the custom call
    function.

    A custom call may apply functions within the scope of the parent module.
    They can be referenced using `called_computations` attribute.

    A custom call can also have layout constraints on operands and results which
    can be specified as optional `operand_layouts` and `result_layouts`
    attributes. The layout attribute is an array of rank-1 index tensors and the
    i-th layout attribute specifies the layout for i-th operand/result.

    The `operand_layouts` & `result_layouts` attributes can be specified under
    the following constraints:
    1) Either both `operand_layouts` and `result_layouts` are specified or none.
    2) None of the operands are of tuple type.
    3) None of the results are of tuple type except the common case of single
       tuple result packing non-tuple values is allowed. In this case the i-th
       `result_layouts` attribute specifies the layout of i-th element in the
       result tuple.

    See https://www.tensorflow.org/xla/operation_semantics#customcall.

    Example:

    ```mlir
    %1 = mhlo.custom_call @foo(%arg0, %arg1) {backend_config = "bar", has_side_effect = true}
          : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$call_target_name,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$has_side_effect,
    OptionalAttr<AnyAttrOf<[StrAttr, DictionaryAttr]>>:$backend_config,
    // TODO(b/189822916): Remove this field when all clients are migrated to
    // the status-returning API.
    DefaultValuedOptionalAttr<
        MHLO_CustomCallApiVersionAttr,
        "::mlir::mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedOptionalAttr<MHLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    DefaultValuedOptionalAttr<MHLO_CustomCallScheduleAttr, "::mlir::mhlo::CustomCallSchedule::NONE">:$custom_call_schedule,
    OptionalAttr<MHLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<MHLO_ArrayOfLayoutAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        TypedArrayAttrBase<
            MHLO_OutputOperandAlias,
            "Aliasing attribute for outputs and operands of CustomCall">,
        "{}">:$output_operand_aliases
  );
  let results = (outs Variadic<MHLO_TensorOrTokenOrTuple>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  // TODO(b/244367323): Need update all usage by adding the arg
  // `output_operand_aliases`, and remove this builder after the bug fix.
  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$result_type, "::mlir::ValueRange":$inputs,
      "::mlir::StringAttr":$call_target_name,
      "::mlir::BoolAttr":$has_side_effect,
      "::mlir::StringAttr":$backend_config,
      "::mlir::mhlo::CustomCallApiVersionAttr":$api_version,
      "::mlir::ArrayAttr":$called_computations,
      "::mlir::ArrayAttr":$operand_layouts,
      "::mlir::ArrayAttr":$result_layouts)>];

  let assemblyFormat = [{
    custom<CustomCallTarget>($call_target_name) `(` $inputs `)`
      attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_DotOp: MHLO_Op<"dot", [Pure]> {
  let summary = "Dot operator";
  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
  let arguments = (
    ins MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs MHLO_Tensor);
  // Dot op required custom exporter to pass the preferred element type
  // to Xla builder.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_DotGeneralOp: MHLO_ShapedInterfaceOp<"dot_general", [Pure]> {
  let summary = "General Dot operator";
  let description = [{
    Performs general dot products between vectors, vector/matrix and
    matrix/matrix multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dotgeneral.
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs MHLO_Tensor);
  let hasCanonicalizer = 1;
  // DotGeneral op required custom exporter to pass the preferred element type
  // to Xla builder.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

// Define Base Einsum op within the HLO dialect as these are client ops and
// therefore this class is not common between HLO and LHLO ops.
class BASE_EinsumOp {
  string summary = "Einsum operator";

  string description = [{
    Returns a tensor whose elements are defined by equation, which is written
    in a shorthand form inspired by the Einstein summation convention.
  }];
}

def MHLO_EinsumOp: MHLO_Op<"einsum", [Pure]>, BASE_EinsumOp {
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );

  let results = (outs MHLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def MHLO_UnaryEinsumOp: MHLO_Op<"unary_einsum", [Pure]>, BASE_EinsumOp {
  let arguments = (ins
    MHLO_Tensor:$operand,
    StrAttr:$einsum_config
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;

  // UnaryEinsumOp is unconditionally canonicalized to the binary EinsumOp so
  // the HLO converter shouldn't be invoked.
  let hasCustomHLOConverter = 1;
}

def MHLO_FftOp: MHLO_Op<"fft", [InferTensorType, Pure]> {
  let summary = "Fast fourier transform operator";
  let description = [{
    Returns the fast-fourier-transform of the input array.

    See
    https://www.tensorflow.org/xla/operation_semantics#fft.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );

  let results = (outs MHLO_Tensor);
}

def MHLO_GatherOp: MHLO_Op<"gather", [InferTensorTypeWithReify, Pure]> {
  let summary = "Gather operator";
  let description = [{
    Stitches together several slices of `operand` from offsets specified in
    `start_indices` (each slice at a potentially different runtime offset).

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_IntTensor:$start_indices,
    MHLO_GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
}

def MHLO_GetDimensionSizeOp: MHLO_Op<"get_dimension_size",
      [Pure, InferTensorType]> {
  let summary = "GetDimensionSize operator";
  let description = [{
    Returns the size of the given dimension of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#getdimensionsize.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64Attr:$dimension
  );
  // TODO(hinsu): Allow 64-bit result types once XLA HLO dialect based on the
  // XLA semantics is available. This limitation is because of the current XLA
  // implementation.
  let results = (outs I32Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def MHLO_MapOp: MHLO_ShapedInterfaceOp<"map",
      [RecursiveMemoryEffects, SameOperandsAndResultShape,
       SingleBlockImplicitTerminator<"ReturnOp">, InferTensorTypeWithReify]> {
  let summary = "Map operator";
  let description = [{
  Applies a scalar function over the given operands arrays, producing an array
  of the same dimensions where each element is the result of the mapped function
  applied to the corresponding elements in the input arrays.

  The mapped function is an arbitrary computation with the restriction that it
  has N inputs of scalar type T and a single output with type S. The output has
  the same dimensions as the operands except that the element type T is replaced
  with S.

  See https://www.tensorflow.org/xla/operation_semantics#map.
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs MHLO_Tensor);
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_ReshapeOp: MHLO_Op<"reshape",
      [Pure, SameOperandsAndResultElementType]> {
  let summary = "Reshape operator";
  let description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.

    Example:

    ```mlir
    %0 = mhlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand);

  let results = (outs MHLO_StaticShapeTensor);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_DynamicReshapeOp: MHLO_ShapedInterfaceOp<"dynamic_reshape", [Pure]> {
  let summary = "Reshape a tensor to a given, possibly dynamic, shape.";
  let description = [{
    Reshapes `operand` to `output_shape`.

    Requires:
    - The length of `output_shape` is equal to the rank of `result`.
    - The number of elements in `operand` (that is, the product of extents of
      its shape) is equal to the number of elements in `output_shape` (that is,
      the product of values in `output_shape`).

    Example:

    ```mlir
    %0 = mhlo.dynamic_reshape %arg0, %shape : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand, MHLO_DimensionTensor:$output_shape);
  let results = (outs MHLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_ScatterOp: MHLO_Op<"scatter",
      [SameVariadicOperandSize, RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Scatter operator";
  let description = [{
    Generates a result which is the value of the input array `operand`,
    with several slices (at indices specified by `scatter_indices`)
    updated with the values in `updates` using `update_computation`.

    See https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    TensorOf<[AnyInteger, Index]>:$scatter_indices,
    Variadic<MHLO_Tensor>:$updates,
    MHLO_ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs Variadic<MHLO_Tensor>);

  let hasCustomHLOConverter = 1;

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def MHLO_SelectOp: MHLO_Op<"select", [Pure, HLO_BroadcastingElementwise,
    InferTensorTypeWithReify]> {
  let summary = "Select operator";
  let description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`. All three operands must be of the same shape
    with the exception of `pred`, which may also be a scalar in which case it is
    broadcasted.

    See https://www.tensorflow.org/xla/operation_semantics#select.

    Example:

    ```mlir
    %0 = mhlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>, tensor<2x3xi32>
    %1 = mhlo.select %arg0, %arg2, %arg3 : (tensor<2x3xi1>, tensor<2x?xi32>, tensor<?x2xi32>) -> tensor<2x?xi32>
    ```
  }];
  let arguments = (ins
    MHLO_PredTensor:$pred,
    MHLO_Tensor:$on_true,
    MHLO_Tensor:$on_false
  );

  let results = (outs MHLO_Tensor:$result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    operands attr-dict `:`
      custom<SelectOpType>(type($pred), type($on_true), type($on_false), type($result))
  }];
}

def MHLO_SelectAndScatterOp: MHLO_Op<"select_and_scatter",
      [RecursiveMemoryEffects, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "SelectAndScatter operator";
  let description = [{
    Runs a windowed selection `select` function over `operand` with shape
    `window_dimensions` and stride `window_strides`. This will produce an amount
    of selected locations whose shape matches `source`. These are then scattered
    to the output which is initialized with `init_value`.
    Multiple scattered elements which land in the same output location are
    combined using the `scatter` function.

    See https://www.tensorflow.org/xla/operation_semantics#selectandscatter.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$source,
    MHLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);

  let results = (outs MHLO_Tensor);

  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_SetDimensionSizeOp: MHLO_Op<"set_dimension_size", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "SetDimensionSize operator";
  let description = [{
    Sets the dynamic size of operand's given dimension. Pass through the operand
    as result, with dynamic dimension tracked by the compiler. Padded values
    will be ignored by downstream reduction ops.

    See https://www.tensorflow.org/xla/operation_semantics#setdimensionsize.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def MHLO_SortOp : MHLO_Op<"sort",
      [RecursiveMemoryEffects, SameOperandsAndResultShape, InferTensorType]> {
  let summary = "Sort operator";
  let description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable
  );

  let results = (outs Variadic<MHLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  // TODO(b/129422361): SortOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

def MHLO_ReverseOp: MHLO_Op<"reverse",
      [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Reverse operator";
  let description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_PartitionIdOp : MHLO_Op<"partition_id", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "PartitionId operator";
  let description = [{
    Returns the value of the partition id of the currently executing device.
    XLA supports two mechanisms for parallel execution: replication and
    partition. A module can be replicated to run on multiple devices
    (replicas) and a module can also be partitioned to split the work
    between devices. replica-id and partition-id returns the id values of the
    current device.

    Example:

    ```mlir
    %1 = mhlo.partition_id : tensor<ui32>
    ```
  }];
  let results = (outs TensorOf<[UI32]>);
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "attr-dict `:` type(results)";
}

def MHLO_PadOp: MHLO_ShapedInterfaceOp<"pad",
      [Pure, SameOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Pad operator";
  let description = [{
    Pads edges and between the elements of `operand` with the `padding_value`
    according to the configuration parameters described below.

    `edge_padding_low` and `edge_padding_high` specify the amount of padding
    added at the low-end (next to index 0) and the high-end (next to the
    highest index) of each dimension respectively. The amount of edge
    padding can be negative -- the absolute value of negative padding indicates
    the number of elements to remove from the specified dimension.

    `interior_padding` specifies the amount of padding (non-negative) added
    between any two elements in each dimension. Interior padding occurs
    logically before edge padding, so in the case of negative edge padding,
    elements are removed from the interior-padded operand.

    This operation is a no-op if, for all dimensions, the edge padding pairs are
    all (0, 0) and the interior padding values are all 0. The figure below shows
    examples of different `edge_padding` and `interior_padding` values for a
    two-dimensional array.

    ![Examples](https://www.tensorflow.org/xla/images/ops_pad.png)

  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$padding_value,
    I64ElementsAttr:$edge_padding_low,
    I64ElementsAttr:$edge_padding_high,
    I64ElementsAttr:$interior_padding
  );

  let results = (outs MHLO_Tensor);

  // TODO(b/129422361): PadOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_TraceOp: MHLO_Op<"trace", []> {
  let summary = "Trace operator";
  let description = [{
    Emits a logging message `tag` with the `operand`.

    Example:

    ```mlir
    mhlo.trace %arg0, "In test code." : tensor<5x1x5xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    StrAttr:$tag
  );
  let hasCustomHLOConverter = 1;
  let assemblyFormat = "$operand `,` $tag attr-dict `:` type($operand)";
}

def MHLO_TransposeOp: MHLO_ShapedInterfaceOp<"transpose",
      [Pure, SameOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def MHLO_TriangularSolveOp: MHLO_Op<"triangular_solve",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "TriangularSolve operator";
  let description = [{
    Solves systems of linear equations with lower or upper triangular
    coefficient matrices by forward- or back-substitution. Broadcasting along
    leading dimensions, this routine solves one of the matrix systems
    op(a) * x = b, or x * op(a) = b, for the variable x, given a and b, where
    op(a) is either op(a) = a, or op(a) = Transpose(a), or
    op(a) = Conj(Transpose(a)).

    Input data is read only from the lower/upper triangle of a, depending on the
    value of lower. Values from the other triangle are ignored. Output data is
    returned in the same triangle; the values in the other triangle are
    implementation-defined and may be anything.

    If the rank of a and b are greater than 2, they are treated as batches of
    matrices, where all except the minor 2 dimensions are batch dimensions. a
    and b must have equal batch dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#triangularsolve.
  }];
  let arguments = (ins
    MHLO_FpOrComplexTensor:$a,
    MHLO_FpOrComplexTensor:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    MHLO_TransposeAttr:$transpose_a
  );
  let results = (outs MHLO_FpOrComplexTensor);
}

def MHLO_ReduceWindowOp: MHLO_Op<"reduce_window", [
      RecursiveMemoryEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType,
    ]> {
  let summary = "ReduceWindow operator";
  let description = [{
    Returns the result of executing a reduction function over all elements in
    each window of one or more arrays in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reducewindow.
  }];

  // TODO(hinsu): Verify that padding attribute is 2-d and the remaining
  // attributes are 1-d. Attributes' leading dimension should match rank of the
  // operands.
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    Variadic<MHLO_Tensor>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<MHLO_Tensor>);

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // Builder for non-variadic version of the operation.
  let builders = [
    OpBuilder<(ins "Type":$result_type, "Value":$operand,
      "Value":$init_value,
      "DenseIntElementsAttr":$window_dimensions,
      "DenseIntElementsAttr":$window_strides,
      "DenseIntElementsAttr":$base_dilations,
      "DenseIntElementsAttr":$window_dilations,
      "DenseIntElementsAttr":$padding),
    [{
      build($_builder, $_state, TypeRange(result_type), ValueRange(operand),
            ValueRange(init_value), window_dimensions, window_strides,
            base_dilations, window_dilations, padding);
    }]>,
    OpBuilder<(ins "ValueRange":$operands,
      "ValueRange":$init_values,
      "DenseIntElementsAttr":$window_dimensions,
      "DenseIntElementsAttr":$window_strides,
      "DenseIntElementsAttr":$base_dilations,
      "DenseIntElementsAttr":$window_dilations,
      "DenseIntElementsAttr":$padding,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$bodyBuilder
    )>,
  ];

  let hasCustomHLOConverter = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
  // TODO(hinsu): Implement custom printer and parser.

  let extraClassDeclaration = [{
     // Get the operation used for reduction applied to `result_index`th result.
     Operation *getReductionOp(int result_index);

     static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
     }
  }];
}

def MHLO_ReturnOp : MHLO_Op<"return", [Pure, Terminator]> {
  let summary = [{
    The `hlo.return` operation terminates a region and returns values.

    Example:

    ```mlir
    %0 = mhlo.reduce %arg0, %arg1 {
      ...
      mhlo.return %1 : tensor<f32>
    }
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple >:$results
  );

  // Disable conversion operator for return op as the op is not an actual XLA
  // instruction and is only used as a terminator for regions.
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

def MHLO_TorchIndexSelectOp : MHLO_Op<"torch_index_select", [Pure]> {
  let summary = "Torch Index Select operator";
  let description = [{
    Returns a new tensor which indexes the input tensor along dimension `dim`
    using the entries in `index`.

    The returned tensor has the same dimensions as `operand`, except for the
    `dim`th dimension which is replaced by the shape of `index` without the
    leading `batch_dims` dimensions;

    The `batch_dims` attribute specifies the number of major batch dimensions
    (0 or more) that act like a multidimensional loop over both the input and
    the index.

    Example:

    ```mlir
    %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
      batch_dims = 1 : i64, dim = 2 : i64
    } : (tensor<8x128x3072x64xf32>, tensor<8x16x1024xi32>) -> tensor<8x128x16x1024x64xf32>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );

  let results = (outs MHLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def MHLO_OptimizationBarrierOp : MHLO_Op<"optimization_barrier",
      [Pure, HLO_PairwiseSameOperandAndResultType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    The `mhlo.optimization_barrier` op blocks optimizations.

    Example:

    ```mlir
    %0:2 = mhlo.optimization_barrier %arg0, %arg1 : (tensor<4x4xf32>, tensor<3x4xf32>) -> (tensor<4x4xf32>, tensor<3x4xf32>)
    ```
  }];

  let description = [{
    Blocks any optimization pass from moving computations across the barrier.

    Ensures that all inputs are evaluated before any operators that depend on the barrier's outputs.
    See
    https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier

    Example:

    ```mlir
    %0:2 = mhlo.optimization_barrier %arg0, %arg1 : tensor<1xf32>, tensor<1xi32>
    %1 = mhlo.optimization_barrier %arg0 : tensor<1xf32>
    mhlo.optimization_barrier()
    ```
  }];

  let arguments = (ins Variadic<MHLO_TensorOrToken>:$operand);

  let results = (outs Variadic<MHLO_TensorOrToken>:$result);

  let hasCustomHLOConverter = 1;

  // Use `attr-dict` before `$operand` because Optional Group anchors in custom
  // directives are currently not supported. Also since inputs are variadic,
  // print `()` if no arguments are present, otherwise parsing is ambiguous:
  //   mhlo.optimization_barrier
  //   %1 = mhlo.add ...
  //   ^ Without lookahead, ambiguous if this is an operand to the previous line
  //     or the start of a separate operation, since newlines are ignored.
  let assemblyFormat = [{
    attr-dict ($operand^ `:` custom<PairwiseOpType>(type($operand), type($result))):(`(` `)`)?
  }];
}

//===----------------------------------------------------------------------===//
// MHLO RNG Operators.
//===----------------------------------------------------------------------===//

def MHLO_RngOp : MHLO_Op<"rng", [InferTensorTypeWithReify, AllElementTypesMatch<["a", "b", "result"]>]> {
  let summary = "RNG with uniform distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the given `rng_distribution` with two parameters:
      `UNIFORM`: the uniform distribution over the interval `[a,b)`. The parameters
                 and output element type have to be a boolean type, an integral type or a
                 floating point types, and the types have to be consistent.

                 See https://www.tensorflow.org/xla/operation_semantics#rnguniform.

      `NORMAL`: the normal distribution with parameters `mu` (=`a`) and
                `sigma` (=`b`). The parameters and output shape have to have a
                floating point elemental type. The parameters furthermore have
                to be scalar valued.

                See https://www.tensorflow.org/xla/operation_semantics#rngnormal.
  }];
  let arguments = (ins
    0DTensorOf<[MHLO_Pred, MHLO_Int, MHLO_Float]>:$a,
    0DTensorOf<[MHLO_Pred, MHLO_Int, MHLO_Float]>:$b,
    MHLO_DimensionTensor:$shape,
    MHLO_RngDistributionAttr:$rng_distribution
  );

  let results = (outs MHLO_PredIntOrFpTensor:$result);

  let hasCustomHLOConverter = 1;
}

def MHLO_RngBitGeneratorOp : MHLO_Op<"rng_bit_generator", [Pure]> {
  let summary = "Uniform random number generator operator";
  let description = [{
    Returns an output with a given shape filled with uniform random bits using
    the specified algorithm (or backend default) and returns an updated state
    (with the same shape as initial state) and the generated random data.

    See https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator.
  }];
  let arguments = (ins
    MHLO_RngAlgorithmAttr:$rng_algorithm,
    MHLO_IntOrFpTensor:$initial_state
  );

  let results = (outs
      MHLO_IntOrFpTensor:$output_state,
      MHLO_StaticShapeIntOrFpTensor:$output
      );

  let hasVerifier = 1;
  // TODO(jpienaar): This should not be needed.
  let hasCustomHLOConverter = 1;
}

def MHLO_XlaRngGetAndUpdateStateOp: MHLO_Op<"xla.rng_get_and_update_state", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "RNG state change";
  let description = [{
    This instruction represents the change of the global random number generator
    state for rng instructions. The global state is incremented by delta and
    the old state is returned.

    The output is currently defined for a single output type. If this changes in
    the future to support multiple types, lowering to use of a global memref
    must ensure that a single memref is still used and updated appropriately.
  }];
  let arguments = (ins I64Attr:$delta);
  let results = (outs StaticShapeTensorOf<[UI64]>);

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";

  // Doesn't have an XLA builder equivalent.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO Quantize Operator.
//===----------------------------------------------------------------------===//

// TODO(b/230662142): Implement unknown scales/zero_point cases.
def MHLO_UniformQuantizeOp : MHLO_UnaryElementwiseOp<"uniform_quantize",
      [Pure], TensorOf<[F32, BF16, MHLO_QuantizedInt]>,
      MHLO_QuantizedIntTensor> {
  let summary = "Uniform quantize operator";
  let description = [{
    Converts floating point tensors or uniform quantized integer tensors to
    uniform quantized integer tensors according to the quantization parameters
    defined by the output type.

    Example:

    ```mlir
    %0 = mhlo.uniform_quantize %arg0 : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    ```
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def MHLO_UniformDequantizeOp : MHLO_UnaryElementwiseOp<"uniform_dequantize",
      [InferTensorType, Pure], MHLO_QuantizedIntTensor, TensorOf<[F32, BF16]>> {
  let summary = "Uniform dequantize operator";
  let description = [{
    Converts quantized array of integers to floating-points according to the
    quantization parameters defined by the input type.

    Example:

    ```mlir
    %0 = mhlo.uniform_dequantize %arg0 : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
    ```
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def MHLO_FusionOp : MHLO_Op<"fusion", []> {
  let summary = "Fusion operator";
  let description = [{
    Models the fusion instruction.

    A fusion op is consists of a group of basic ops (represented as a region
    attached to it). It serves as a hint to the backend that it is beneficial
    to emit the contained ops into a single loop nest or kernel.
  }];
  let regions = (region SizedRegion<1>:$fused_computation);

  let arguments = (ins
    Variadic<MHLO_TensorOrToken>:$inputs,
    OptionalAttr<MHLO_FusionKindAttr>:$fusion_kind,
    DefaultValuedOptionalAttr<
        TypedArrayAttrBase<
            MHLO_OutputOperandAlias,
            "Aliasing attribute for outputs and operands of Fusion">,
        "{}">:$output_operand_aliases
  );

  let results = (outs
    Variadic<AnyTypeOf<[MHLO_Tensor, MHLO_Tuple]>>:$results
  );

  // FusionOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasVerifier = 1;
}

// This is an op for purposes internal to XLA/GPU.
def MHLO_BitcastOp : MHLO_Op<"bitcast", [Pure]> {
  let summary = "Bitcast operator";
  let description = [{
    This op changes the shape of the input in the way that the physical
    arrangement of elements are unchanged.

    However, the op needs layout information to make sense of "physical
    arrangement of elements". Layout support in MHLO is currently under
    exploration.

    Example:

    ```mlir
    %0 = mhlo.bitcast %arg0 : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
    ```
  }];

  let arguments = (ins MHLO_Tensor:$operand);
  let results = (outs MHLO_Tensor);
  let hasCustomHLOConverter = 1;
  let hasFolder = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_ReducePrecisionOp :
    MHLO_Op<"reduce_precision", [HLO_CompatibleOperandsAndResultType, Pure]> {
  let summary = "Reduce precision operator";
  let description = [{
    Models the effect of converting floating - point values to a lower -
    precision format(such as IEEE - FP16) and back to the original
    format. The number of exponent and mantissa bits in the lower -
    precision format can be specified arbitrarily,
    although all bit sizes may not be supported on all hardware
    implementations.

    See https://www.tensorflow.org/xla/operation_semantics#reduceprecision.

    ```mlir
    %0 = mhlo.reduce_precision %arg0, format = e8m10 : tensor<3x4xf32>
    ```
  }];
  let arguments = (ins
    MHLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let hasVerifier = 1;
  let results = (outs MHLO_FpTensor:$output);

  let assemblyFormat = [{
    $operand `,` `format` `=` custom<ExponentMantissa>($exponent_bits, $mantissa_bits)
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($output))
  }];
}

def MHLO_RealDynamicSliceOp: MHLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "Real Dynamic Slice operator";
  let description = [{
    The dynamic shape version of SliceOp. Extracts a sub-array from the input
    array according to start_indices, limit_indices and strides. Expect
    start_indices/limit_indices/strides to be statically shaped and matching
    the rank of the input.

    Example:

    ```mlir
    %0 = mhlo.real_dynamic_slice %input, %start, %limit, %strides
           : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_DimensionTensor:$start_indices,
    MHLO_DimensionTensor:$limit_indices,
    MHLO_DimensionTensor:$strides
  );
  let results = (outs MHLO_Tensor:$result);
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_DynamicPadOp: MHLO_ShapedInterfaceOp<"dynamic_pad",
      [Pure, AllElementTypesMatch<["operand", "padding_value", "result"]>,
      AllTypesMatch<["edge_padding_low", "edge_padding_high", "interior_padding"]>]> {
  let summary = "Dynamic Pad operator";
  let description = [{
    The dynamic shape version of PadOp. Pads the edges of `operand` with the
    `padding_value` and according to the passed configuration. Expect
    edge_padding_low/edge_padding_high/interior_padding to be statically shaped
    and matching the rank of the input.

    See https://www.tensorflow.org/xla/operation_semantics#pad.

    Example:

    ```mlir
    %0 = mhlo.dynamic_pad %arg0, %arg1, %arg2, %arg3, %arg4
           : (tensor<?x?xf32>, tensor<f32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<?x?xf32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_Tensor:$padding_value,
    MHLO_DimensionTensor:$edge_padding_low,
    MHLO_DimensionTensor:$edge_padding_high,
    MHLO_DimensionTensor:$interior_padding
  );
  let results = (outs MHLO_Tensor:$result);
  let description = [{
    Dynamically Pads the `operand`, with amount of padding added at
    low-end/high-end/interior is passed through input tensors.
  }];
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_DynamicGatherOp: MHLO_Op<"dynamic_gather",
                                [InferTensorTypeWithReify, Pure]> {
  string summary = "Dynamic Gather operator";
  string description = [{
    The dynamic shape version of GatherOp. Stitches together several slices of
    an input array.
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_IntTensor:$start_indices,
    MHLO_IntTensor:$slice_sizes,
    MHLO_GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs MHLO_Tensor);

  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
}

def MHLO_DynamicConvOp : MHLO_Op<"dynamic_conv", [Pure]> {
  let summary = "Dynamic Convolution operator";
  let description = [{
    The dynamic shape version of ConvOp. Computes a convolution with dynamic padding.
  }];

  let arguments = !con(
    (ins
       MHLO_Tensor:$lhs,
       MHLO_Tensor:$rhs,
       MHLO_Tensor:$d_padding),
    MHLO_ConvolutionAttributes.attributes);
  let results = (outs MHLO_Tensor);
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_ComputeReshapeShapeOp :
    MHLO_Op<"compute_reshape_shape", [Pure]> {
  string summary = "Compute input for reshape with any dynamic dim resolved";

  string description = [{
    This operation handles the dynamic aspect of a TF/NumPy/CHLO reshape. The
    dynamic aspect is that a single extent can be -1 and that dimension will
    instead be computed. This handles the computation and can then be passed to
    an HLO DynamicReshapeOp to replicate the TF/NumPy reshape behavior.

    This op has undefined behavior if the dimensions do not evenly divide the
    number of elements, or if there are multiple -1 values. It is an identity op
    if no dimensions are -1.

    ```
    %0 = hlo.compute_reshape_shape 12, [2, -1] -> [2, 6]
    ```

    Example:

    ```mlir
    %0 = mhlo.compute_reshape_shape %arg0, %arg1 : (index, tensor<2xi32>) -> tensor<2xi32>
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_CstrReshapableOp :
    MHLO_Op<"cstr_reshapable", [Pure]> {
  string summary = "Compute input for reshape with any dynamic dim resolved";

  string description = [{
    This operation creates a witness on the constraint that a given shape would
    be a valid reshape for the given number of elements.

    ```
    %0 = mhlo.cstr_reshapable 12, [2, -1] -> success
    %1 = mhlo.cstr_reshapable 13, [2, -1] -> failure
    ```

    Example:

    ```mlir
    %0 = mhlo.cstr_reshapable %arg0, %arg1 : (index, tensor<3xi32>) -> !shape.witness
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs Shape_WitnessType:$result);

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS
