/* Copyright 2019 The OpenXLA Authors.

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
include "mlir/IR/SymbolInterfaces.td"
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
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    Example:
    ```mlir
    %output = mhlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    ```
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
  let summary = "Iota operation";
  let description = [{
    Fills an `output` tensor with values in increasing order starting from zero
    along the `iota_dimension` dimension.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %output = mhlo.iota dim = 0 : tensor<4x5xi32>
    ```
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
  let summary = "DynamicIota operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as IotaOp except that the
    result shape is specified dynamically via `output_shape`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %0 = mhlo.dynamic_iota %arg0, dim = 0 : (tensor<1xindex>) -> tensor<4xi32>
    ```
  }];

  let arguments = (ins MHLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs MHLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}


def MHLO_CreateTokenOp : MHLO_Op<"create_token", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "CreateToken operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AfterAllOp with 0 inputs:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %output = mhlo.create_token : !mhlo.token
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
     RankedTensorOf<[MHLO_SInt, MHLO_Float, MHLO_Complex, MHLO_QuantizedInt]>,
     RankedTensorOf<[MHLO_SInt, MHLO_Float, MHLO_QuantizedInt]>> {
  let summary = "Abs operation";
  let description = [{
    Performs element-wise abs operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

    Example:
    ```mlir
    %result = mhlo.abs %operand : tensor<3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_CbrtOp: MHLO_UnaryElementwiseOp<"cbrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Cbrt operation";
  let description = [{
    Performs element-wise cubic root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cbrt

    Example:
    ```mlir
    %result = mhlo.cbrt %operand : tensor<4xf32>
    ```
  }];
}
def MHLO_CeilOp: MHLO_UnaryElementwiseOp<"ceil",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Ceil operation";
  let description = [{
    Performs element-wise ceil of `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#ceil

    Example:
    ```mlir
    %result = mhlo.ceil %operand : tensor<5xf32>
    ```
  }];
}
def MHLO_ConvertOp : MHLO_UnaryElementwiseOp<"convert",
    [Pure, SameOperandsAndResultShape], MHLO_Tensor> {
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = mhlo.convert %operand : (tensor<3xi32>) -> tensor<3xcomplex<f32>>
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
  let summary = "Clz operation";
  let description = [{
    Performs element-wise count of the number of leading zero bits in the
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

    Example:
    ```mlir
    %result = mhlo.count_leading_zeros %operand : tensor<2x2xi8>
    ```
  }];
}

def MHLO_CosineOp: MHLO_UnaryElementwiseOp<"cosine",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Cosine operation";
  let description = [{
    Performs element-wise cosine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cosine

    Example:
    ```mlir
    %result = mhlo.cosine %operand : tensor<2xf32>
    ```
  }];

  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_ErfOp: MHLO_UnaryElementwiseOp<"erf",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Erf operation";
  let description = [{
    Performs element-wise erf operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#erf

    Example:
    ```mlir
    %result = mhlo.erf %operand : tensor<2x2xf32>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_ExpOp: MHLO_UnaryElementwiseOp<"exponential",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Exp operation";
  let description = [{
    Performs element-wise exponential operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential

    Example:
    ```mlir
    %result = mhlo.exponential %operand : tensor<2x2xf64>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_Expm1Op: MHLO_UnaryElementwiseOp<"exponential_minus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Expm1 operation";
  let description = [{
    Performs element-wise exponential minus one operation on `operand` tensor
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#exponential_minus_one

    Example:
    ```mlir
    %result = mhlo.exponential_minus_one %operand : tensor<2xf32>
    ```
  }];
}
def MHLO_FloorOp: MHLO_UnaryElementwiseOp<"floor",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Floor operation";
  let description = [{
    Performs element-wise floor of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#floor

    Example:
    ```mlir
    %result = mhlo.floor %operand : tensor<2xf32>
    ```
  }];
}
def MHLO_ImagOp: MHLO_UnaryElementwiseOp<"imag",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    MHLO_FpOrComplexTensor, MHLO_FpTensor> {
  let summary = "Imag operation";
  let description = [{
    Extracts the imaginary part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#imag

    Example:
    ```mlir
    %result = mhlo.imag %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
  }];

  let hasFolder = 1;

}

def MHLO_IsFiniteOp: MHLO_UnaryElementwiseOp<"is_finite", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>], MHLO_Tensor> {
  let summary = "IsFinite operation";
  let description = [{
    Performs element-wise check whether the value in `x` is finite (i.e. is
    neither +Inf, -Inf, nor NaN) and produces a `y` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#is_finite

    Example:
    ```mlir
    %y = mhlo.is_finite %x : (tensor<7xf32>) -> tensor<7xi1>
    ```
  }];
  let arguments = (ins MHLO_FpTensor:$x);
  let results = (outs MHLO_PredTensor:$y);

  let assemblyFormat = [{
    $x attr-dict `:` functional-type(operands, results)
  }];
}

def MHLO_LogOp: MHLO_UnaryElementwiseOp<"log",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Log operation";
  let description = [{
    Performs element-wise logarithm operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log

    Example:
    ```mlir
    %result = mhlo.log %operand : tensor<2x2xf64>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_Log1pOp: MHLO_UnaryElementwiseOp<"log_plus_one",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Log1p operation";
  let description = [{
    Performs element-wise logarithm plus one operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#log_plus_one

    Example:
    ```mlir
    %result = mhlo.log_plus_one %operand : tensor<6xf32>
    ```
  }];
}
def MHLO_LogisticOp: MHLO_UnaryElementwiseOp<"logistic",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Logistic operation";
  let description = [{
    Performs element-wise logistic operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#logistic

    Example:
    ```mlir
    %result = mhlo.logistic %operand : tensor<2x2xf32>
    ```
  }];
  let hasFolder = 1;
}
def MHLO_NotOp: MHLO_UnaryElementwiseOp<"not",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_PredOrIntTensor> {
  let summary = "Not operation";
  let description = [{
    Performs element-wise NOT of tensor `operand` of type integer and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not

    Example:
    ```mlir
    %result = mhlo.not %operand : tensor<5x3x1xi1>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_NegOp: MHLO_UnaryElementwiseOp<"negate",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexOrQuantizedIntTensor> {
  let summary = "Neg operation";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = mhlo.negate %operand : tensor<2x3xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PopulationCountOp: MHLO_UnaryElementwiseOp<"popcnt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "PopulationCount operation";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand`
    tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

    Example:
    ```mlir
    %result = mhlo.popcnt %operand : tensor<4xi8>
    ```
  }];
}
def MHLO_RealOp: MHLO_UnaryElementwiseOp<"real",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    MHLO_FpOrComplexTensor, MHLO_FpTensor> {
  let summary = "Real operation";
  let description = [{
    Extracts the real part, element-wise, from the `operand` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#real

    Example:
    ```mlir
    %result = mhlo.real %operand : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RoundOp: MHLO_UnaryElementwiseOp<"round_nearest_afz",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "Round operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    away from zero, on the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_afz

    Example:
    ```mlir
    %result = mhlo.round_nearest_afz %operand : tensor<5xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RoundNearestEvenOp: MHLO_UnaryElementwiseOp<"round_nearest_even",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpTensor> {
  let summary = "RoundNearestEven operation";
  let description = [{
    Performs element-wise rounding towards the nearest integer, breaking ties
    towards the even integer, on the `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#round_nearest_even

    Example:
    ```mlir
    %result = mhlo.round_nearest_even %operand : tensor<5xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_RsqrtOp: MHLO_UnaryElementwiseOp<"rsqrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Rsqrt operation";
  let description = [{
    Performs element-wise reciprocal square root operation on `operand` tensor
    and produces a `result` tensor, implementing the `rSqrt` operation from the
    IEEE-754 specification.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rsqrt

    Example:
    ```mlir
    %result = mhlo.rsqrt %operand : tensor<2x2xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_SignOp: MHLO_UnaryElementwiseOp<"sign",
    [Pure, HLO_CompatibleOperandsAndResultType],
    RankedTensorOf<[MHLO_SInt, MHLO_Float, MHLO_Complex, HLO_QuantizedInt]>> {
  let summary = "Sign operation";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = mhlo.sign %operand : tensor<7xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_SineOp: MHLO_UnaryElementwiseOp<"sine",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Sine operation";
  let description = [{
    Performs element-wise sine operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sine

    Example:
    ```mlir
    %result = mhlo.sine %operand : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_TanOp: MHLO_UnaryElementwiseOp<"tan",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpOrComplexTensor> {
  let summary = "Tan operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/954.

    Informally, this operation returns `Tan(operand)` element-wise.

    Example:
    ```mlir
    %0 = mhlo.tan %arg0 : tensor<2xf32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_SqrtOp: MHLO_UnaryElementwiseOp<"sqrt",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Sqrt operation";
  let description = [{
    Performs element-wise square root operation on `operand` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sqrt

    Example:
    ```mlir
    %result = mhlo.sqrt %operand : tensor<2x2xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_TanhOp: MHLO_UnaryElementwiseOp<"tanh",
    [Pure, HLO_CompatibleOperandsAndResultType],
    MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Tanh operation";
  let description = [{
    Performs element-wise hyperbolic tangent operation on `operand` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tanh

    Example:
    ```mlir
    %result = mhlo.tanh %operand : tensor<2xf32>
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
  let summary = "Add operation";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

    Example:
    ```mlir
    %result = mhlo.add %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_Atan2Op : MHLO_BinaryElementwiseOp<"atan2",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_FpComplexOrQuantizedIntTensor> {
  let summary = "Atan2 operation";
  let description = [{
    Performs element-wise atan2 operation on `lhs` and `rhs` tensor and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#atan2

    Example:
    ```mlir
    %result = mhlo.atan2 %lhs, %rhs : tensor<3xf32>
    ```
  }];
}

def MHLO_ComplexOp: MHLO_BinaryElementwiseOp<"complex", [Pure,
    SameOperandsElementType, SameOperandsAndResultShape,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Complex operation";
  let description = [{
    Performs element-wise conversion to a complex value from a pair of real and
    imaginary values, `lhs` and `rhs`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#complex

    Example:
    ```mlir
    %result = mhlo.complex %lhs, %rhs : tensor<2xcomplex<f32>>
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
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexOrQuantizedIntTensor> {
  let summary = "Div operation";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = mhlo.divide %lhs, %rhs : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MaxOp : MHLO_BinaryElementwiseOp<"maximum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Max operation";
  let description = [{
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = mhlo.maximum %lhs, %rhs : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MinOp : MHLO_BinaryElementwiseOp<"minimum",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Min operation";
  let description = [{
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = mhlo.minimum %lhs, %rhs : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MulOp : MHLO_BinaryElementwiseOp<"multiply",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Mul operation";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

    Example:
    ```mlir
    %result = mhlo.multiply %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PowOp : MHLO_BinaryElementwiseOp<"power",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexOrQuantizedIntTensor> {
  let summary = "Pow operation";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = mhlo.power %lhs, %rhs : tensor<6xf32>
    ```
  }];
}
def MHLO_RemOp : MHLO_BinaryElementwiseOp<"remainder",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexOrQuantizedIntTensor> {
  let summary = "Rem operation";
  let description = [{
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = mhlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_ShiftLeftOp : MHLO_BinaryElementwiseOp<"shift_left",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftLeft operation";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

    Example:
    ```mlir
    %result = mhlo.shift_left %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_ShiftRightArithmeticOp : MHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftRightArithmetic operation";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor
    by `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

    Example:
    ```mlir
    %result = mhlo.shift_right_arithmetic %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_ShiftRightLogicalOp : MHLO_BinaryElementwiseOp<"shift_right_logical",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "ShiftRightLogical operation";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

    Example:
    ```mlir
    %result = mhlo.shift_right_logical %lhs, %rhs : tensor<6xi8>
    ```
  }];
}

def MHLO_SubtractOp : MHLO_BinaryElementwiseOp<"subtract",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntFpOrComplexOrQuantizedIntTensor> {
  let summary = "Subtract operation";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

    Example:
    ```mlir
    %result = mhlo.subtract %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_StochasticConvertOp : MHLO_Op<"stochastic_convert",
      [Pure, Elementwise, AllShapesMatch<["operand", "random", "result"]>]> {
  let summary = "StochasticConvert operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/295.

    Informally, this operation performs element-wise conversion of values from
    a bigger type to a smaller one with stochastic rounding using the random
    number passed in.
  }];

  let arguments = (ins MHLO_FpTensor:$operand, RankedTensorOf<[MHLO_UInt]>:$random);
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
  let summary = "And operation";
  let description = [{
    Performs element-wise AND of two tensors `lhs` and `rhs` and produces a
    `result` tensor

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and

    Example:
    ```mlir
    %result = mhlo.and %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
}

def MHLO_OrOp: MHLO_BinaryBiwiseOrLogicalElementwiseOp<"or"> {
  let summary = "Or operation";
  let description = [{
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or

    Example:
    ```mlir
    %result = mhlo.or %lhs, %rhs : tensor<2xi1>
    ```
  }];
}

def MHLO_XorOp : MHLO_BinaryBiwiseOrLogicalElementwiseOp<"xor"> {
  let summary = "Xor operation";
  let description = [{
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor

    Example:
    ```mlir
    %result = mhlo.xor %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MHLO communication op definitions.
//===----------------------------------------------------------------------===//

// InfeedOp corresponds to 'InfeedWithToken' xla client API and not 'Infeed'.
// InfeedWithToken allows ordering of infeed HLO instructions using tokens.
def MHLO_InfeedOp : MHLO_Op<"infeed", []> {
  let summary = "Infeed operation";
  let description = [{
    Reads data from the infeed and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#infeed

    Example:
    ```mlir
    %results:2 = "mhlo.infeed"(%token) {
      infeed_config = ""
    } : (!mhlo.token) -> (tensor<3x3x3xi32>, !mhlo.token)
    ```
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
  let summary = "Outfeed operation";
  let description = [{
    Writes `inputs` to the outfeed and produces a `result` token.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#outfeed

    Example:
    ```mlir
    %result = "mhlo.outfeed"(%input0, %token) {
      outfeed_config = ""
    } : (tensor<3x3x3xi32>, !mhlo.token) -> !mhlo.token
    ```
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
  let summary = "Send operation";
  let description = [{
    Sends `inputs` to a channel `channel_id` and produces a `result` token.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#send

    Example:
    ```mlir
    %result = "mhlo.send"(%operand, %token) {
      // channel_id = 5 : i64,
      // channel_type = #stablehlo<channel_type DEVICE_TO_HOST>,
      channel_handle = #mhlo.channel_handle<handle = 5, type = 2>,
      is_host_transfer = true
    } : (tensor<3x4xi32>, !mhlo.token) -> !mhlo.token
    ```
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
  let summary = "Recv operation";
  let description = [{
    Receives data from a channel with `channel_id` and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#recv

    Example:
    ```mlir
    %results:2 = "mhlo.recv"(%token) {
      // channel_id = 5 : i64,
      // channel_type = #stablehlo<channel_type HOST_TO_DEVICE>,
      channel_handle = #mhlo.channel_handle<handle = 5, type = 3>,
      is_host_transfer = true
    } : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
    ```
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
  let summary = "ReplicaId operation";
  let description = [{
    Produces `replica_id` of the current process.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#replica_id

    Example:
    ```mlir
    %result = mhlo.replica_id : tensor<ui32>
    ```
  }];
  let results = (outs UI32RankedTensor);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// MHLO control flow op definitions.
//===----------------------------------------------------------------------===//

def MHLO_AddDependencyOp : MHLO_Op<"add_dependency", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AddDependency operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation two operands: a data operand and a token. The
    output of the operation is the data operand. When used with AfterAll this
    operation enables ordering non-side-effecting operations (those that do not
    produce token values).

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
  let summary = "AfterAll operation";
  let description = [{
    Ensures that the operations producing the `inputs` are executed before any
    operations that depend on `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %result = mhlo.after_all %input0, %input1 : !mhlo.token
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
  let summary = "AsyncStart operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation kicks off an asynchronous computation.

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

    This returns all the state needed between async ops. After buffer
    assignment, the return values represents the space needed to hold the input,
    results, and any scratchpads needed or edited by the async op.
  }];

  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple>:$inputs,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread
  );

  let results = (outs MHLO_AsyncBundle);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_AsyncUpdateOp : MHLO_Op<"async_update", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AsyncUpdate operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation blocks on an asynchronous computation until a
    sync barrier. This returns `bundle` after operating on it.

    See the documentation for AsyncStart for more information.
  }];

  let arguments = (ins
    MHLO_AsyncBundle:$bundle,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread
    );

  let results = (outs MHLO_AsyncBundle);

  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_AsyncDoneOp : MHLO_Op<"async_done", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AsyncDone operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation blocks until the end of an asynchronous computation.
    It returns the final result of the asynchronous computation.

    See the documentation for AsyncStart for more information.
  }];

  let arguments = (ins
    MHLO_AsyncBundle:$bundle,
    FlatSymbolRefAttr:$called_computation,
    StrAttr:$execution_thread
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
  let summary = "If operation";
  let description = [{
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "mhlo.if"(%pred) ({
      "mhlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "mhlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
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
  let summary = "Case operation";
  let description = [{
    Produces the output from executing exactly one `function` from `branches`
    depending on the value of `index`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case

    Example:
    ```mlir
    %result0, %result1 = "mhlo.case"(%index) ({
      mhlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
    }, {
      mhlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    ```
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
  let summary = "While operation";
  let description = [{
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = "mhlo.while"(%operand0, %operand1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction LT>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %constant0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    ```
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
  string summary = "AllGather operation";
  string description = [{
    Within each process group in the process grid, concatenates the values of the
    operand tensor from each process along `all_gather_dim` and produces a
    result tensor. The `computation` is applied separately for each operand in
    `operands`, producing one result per operand.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_gather

    Example:
    ```mlir
    %result = "mhlo.all_gather"(%operand) {
      all_gather_dim = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
      // channel_id = 0
      channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
      // use_global_device_ids = false
    } : (tensor<2x2xf32>) -> tensor<2x4xf32>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$operands,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let results = (outs Variadic<MHLO_Tensor>);
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

def MHLO_AllReduceOp : MHLO_Op<"all_reduce", [
    SingleBlockImplicitTerminator<"ReturnOp">,
    InferTensorType
  ]> {
  let summary = "AllReduce operation";
  let description = [{
    Within each process group in the process grid, applies a reduction function
    `computation` to the values of an operand tensor from each process and
    produces a result tensor. The `computation` is applied separately for each
    operand in `operands`, producing one result per operand.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

    Example:
    ```mlir
    %result = "mhlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = mhlo.add %arg1, %arg2 : tensor<f32>
        mhlo.return %0 : tensor<f32>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
      // channel_id = 0
      channel_handle = #mhlo.channel_handle<handle = 0, type = 0>
      // use_global_device_ids = false
    } : (tensor<4xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$operands,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle,
    UnitAttr:$use_global_device_ids
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs Variadic<MHLO_Tensor>);

  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$replica_groups,
      "::mlir::mhlo::ChannelHandleAttr":$channel_handle,
      CArg<"bool", "false">:$use_global_device_ids)>,
     OpBuilder<(ins
      "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$replica_groups,
      "::mlir::mhlo::ChannelHandleAttr":$channel_handle,
      CArg<"bool", "false">:$use_global_device_ids)>,
 ];

  let hasCustomHLOConverter = 1;
}

def MHLO_ReduceScatterOp : MHLO_Op<"reduce_scatter", []> {
  let summary = "ReduceScatter operation";
  let description = [{
     Within each process group in the process grid, performs reduction, using
     `computations`, over the values of the `operand` tensor from each process,
     splits the reduction result along `scatter_dimension` into parts, and
     scatters the split parts between the processes to produce the `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_scatter

    Example:
    ```mlir
    %result = "mhlo.reduce_scatter"(%operand) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = mhlo.add %arg0, %arg1 : tensor<f32>
      mhlo.return %0 : tensor<f32>
    }) {
      scatter_dimension = 1 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      // channel_id = 0
      channel_handle = #mhlo.channel_handle<handle = 0, type = 0>
      // use_global_device_ids = false
    } : (tensor<2x4xf32>) -> tensor<2x2xf32>
    ```
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
  let summary = "AllToAll operation";
  let description = [{
    Within each process group in the process grid, splits the values of the
    `operand` tensor along `split_dimension` into parts, scatters the split parts
    between the processes, concatenates the scattered parts along `concat_dimension`
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_to_all

    Example:
    ```mlir
    %result = "mhlo.all_to_all"(%operand) {
      split_dimension = 1 : i64,
      concat_dimension = 0 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<2x4xf32>) -> tensor<4x2xf32>
    ```
  }];

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
  let summary = "Reduce operation";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "mhlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
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

  // Builder
  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$init_values,
      "DenseIntElementsAttr":$dimensions, "TypeRange":$element_types)>,
  ];

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def MHLO_GetTupleElementOp: MHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "GetTupleElement operation";
  let description = [{
    Extracts element at `index` position of the `operand` tuple and produces a
    `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

    Example:
    ```mlir
    %result = mhlo.get_tuple_element %operand[0] : (tuple<tensor<2xf32>, tuple<tensor<i32>>>) -> tensor<2xf32>
    ```
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
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = mhlo.tuple %val0, %val1 : tuple<tensor<2xf32>, tuple<tensor<i32>>>
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
  let summary = "Compare operation";
  let description = [{
    Performs element-wise comparison of `lhs` and `rhs` tensors according to
    `comparison_direction` and `compare_type`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare

    Example:
    ```mlir
    %result = mhlo.compare LT, %lhs, %rhs, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
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
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = "mhlo.slice" (%operand) {
      start_indices = dense<[1, 2]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi64>) -> tensor<2x2xi64>
    ```
  }];
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
  let summary = "DynamicSlice operation";
  let description = [{
    Extracts a slice from the `operand` using dynamically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = mhlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
      : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
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
  let summary = "DynamicUpdateSlice operation";
  let description = [{
    Produces a `result` tensor which is equal to the `operand` tensor except
    that the slice starting at `start_indices` is updated with the values in
    `update`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice

    Example:
    ```mlir
    %result = mhlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1
      : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
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
  let summary = "Domain operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, these operations are used to group instructions with the same
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
  let summary = "BatchNormGrad operation";
  let description = [{
    Computes gradients of several inputs of BatchNormTrainingOp backpropagating
    from `grad_output`, and produces `grad_operand`, `grad_scale` and
    `grad_offset` tensors.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_grad

    Example:
    ```mlir
    %grad_operand, %grad_scale, %grad_offset =
    "mhlo.batch_norm_grad"(%operand, %scale, %mean, %variance, %grad_output) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
        tensor<2x2x2xf32>) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    ```
  }];

  let arguments = (ins
    MHLO_FpTensor:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$mean,
    1DTensorOf<[MHLO_Float]>:$variance,
    MHLO_FpTensor:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      MHLO_FpTensor:$grad_operand,
      1DTensorOf<[MHLO_Float]>:$grad_scale,
      1DTensorOf<[MHLO_Float]>:$grad_offset);

  let hasCustomHLOConverter = 1;
}

def MHLO_BatchNormInferenceOp : MHLO_Op<"batch_norm_inference",
    [Pure, AllElementTypesMatch<["operand", "result"]>, InferTensorType]> {
  let summary = "BatchNormInference operation";
  let description = [{
    Normalizes the `operand` tensor across all dimensions except for the
    `feature_index` dimension and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_inference

    Example:
    ```mlir
    %result = "mhlo.batch_norm_inference"(%operand, %scale, %offset, %mean, %variance) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x2xf32>
    ```
  }];

  let arguments = (ins
    MHLO_FpTensor:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$offset,
    1DTensorOf<[MHLO_Float]>:$mean,
    1DTensorOf<[MHLO_Float]>:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs MHLO_FpTensor:$result);
}

def MHLO_BatchNormTrainingOp : MHLO_Op<"batch_norm_training",
    [Pure, AllElementTypesMatch<["operand", "output", "batch_mean", "batch_var"]>,
    InferTensorType]> {
  let summary = "BatchNormTraining operation";
  let description = [{
    Computes mean and variance across batch and spatial dimensions and
    normalizes the `operand` tensor, for each feature in the `feature_index`
    dimension and produces `output`, `batch_mean` and `batch_var` tensors.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_training

    Example:
    ```mlir
    %output, %batch_mean, %batch_var = "mhlo.batch_norm_training"(%operand, %scale, %offset) {
      epsilon = 0.0 : f32,
      feature_index = 2 : i64
    } : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    ```
  }];

  let arguments = (ins
    MHLO_FpTensor:$operand,
    1DTensorOf<[MHLO_Float]>:$scale,
    1DTensorOf<[MHLO_Float]>:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      MHLO_FpTensor:$output,
      1DTensorOf<[MHLO_Float]>:$batch_mean,
      1DTensorOf<[MHLO_Float]>:$batch_var);

  let hasCustomHLOConverter = 1;
}

def MHLO_BitcastConvertOp : MHLO_ShapedInterfaceOp<"bitcast_convert",
    [Pure]> {
  let summary = "BitcastConvert operation";
  let description = [{
    Performs a bitcast operation on `operand` tensor and produces a `result`
    tensor where the bits of the entire `operand` tensor are reinterpreted using
    the type of the `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

    Example:
    ```mlir
    %result = mhlo.bitcast_convert %operand : (tensor<2xf32>) -> tensor<2x4xi8>
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
  let summary = "Broadcast operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as XLA's Broadcast:
    https://www.tensorflow.org/xla/operation_semantics#broadcast

    Example:
    ```mlir
    %result = mhlo.broadcast %operand, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
}

def MHLO_BroadcastInDimOp : MHLO_Op<"broadcast_in_dim",
      [Pure, HLO_CompatibleOperandsAndResultElementType]> {
  let summary = "BroadcastInDim operation";
  let description = [{
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = mhlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
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
  let summary = "DynamicBroadcastInDim operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as BroadcastInDimOp except
    that the result shape is specified dynamically via `output_dimensions`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    It also accepts optional attributes to express static knowledge about the
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
  let summary = "Cholesky operation";
  let description = [{
    Computes the Cholesky decomposition of a batch of matrices.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#cholesky

    Example:
    ```mlir
    %result = mhlo.cholesky %a, lower = true : tensor<3x3xf32>
    ```
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
  let summary = "Clamp operation";
  let description = [{
    Clamps every element of the `operand` tensor between a minimum and maximum
    value and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp

    Example:
    ```mlir
    %result = mhlo.clamp %min, %operand, %max : tensor<3xi32>
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
  let summary = "Concatenate operation";
  let description = [{
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = mhlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_Tensor>:$val,
    I64Attr:$dimension
  );

  let results = (outs MHLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def MHLO_CollectiveBroadcastOp: MHLO_Op<"collective_broadcast",
    [HLO_CompatibleOperandsAndResultType]> {
  let summary = "CollectiveBroadcast operation";
  let description = [{
    Within each process group in the process grid, send the value of the
    `operand` tensor from the source process to the target processes and produce a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_broadcast

    Example:
    ```mlir
    %result = "mhlo.collective_broadcast"(%operand) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #mhlo.channel_handle<handle = 0, type = 0>
    } : (tensor<1x2xi64>) -> tensor<1x2xi64>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<MHLO_ChannelHandle>:$channel_handle
  );
  let results = (outs MHLO_Tensor);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
  // channel_handle is only used for the SPMD partitioner, so we add a
  // simplified builder method for convenience.
  let builders = [
    OpBuilder<(ins
      "::mlir::Type":$result_type, "::mlir::Value":$operand,
      "::mlir::DenseIntElementsAttr":$replica_groups)>];
}

def MHLO_CollectivePermuteOp: MHLO_Op<"collective_permute",
    [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "CollectivePermute operation";
  let description = [{
    Within each process group in the process grid, sends the value of the
    `operand` tensor from the source process to the target process and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#collective_permute

    Example:
    ```mlir
    %result = "mhlo.collective_permute"(%operand) {
      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
      // channel_id = 0
      channel_handle = #mhlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4x2xf32>) -> tensor<4x2xf32>
    ```
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

def MHLO_CompositeOp : MHLO_Op<"composite", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Composite operation";
  let description = [{
    Encapsulates an operation made up (composed) of other StableHLO operations,
    taking `inputs` and `composite_attributes` and producing `results`. The
    semantics of the op are implemented by the `decomposition` attribute. The
    `composite` op can be replaced with its decomposition without changing program
    semantics. In cases where inlining the decomposition does not provide the same
    op semantics, prefer using `custom_call`.

    The `version` field (defaults to `0`) is used to denote when a composite's
    semantics change.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite

    Example:
    ```mlir
    %results = mhlo.composite "my.op" %arg0, %arg1 {
      decomposition = @my_op,
      composite_attributes = { my_attribute = "my_value" },
      version = 1 : i32
    } : (tensor<f32>, tensor<f32>) -> tensor<f32>
    ```
  }];

  let arguments = (ins
    Variadic<MHLO_TensorOrTokenOrTuple>:$inputs,
    StrAttr:$name,
    DefaultValuedOptionalAttr<DictionaryAttr, "{}">:$composite_attributes,
    FlatSymbolRefAttr:$decomposition,
    DefaultValuedOptionalAttr<I32Attr, "0">:$version
  );
  let results = (outs Variadic<MHLO_TensorOrTokenOrTuple>);
  let hasCustomHLOConverter = 1;
  let assemblyFormat = "$name $inputs attr-dict `:` functional-type(operands, results)";
}

def MHLO_ConvolutionOp : MHLO_Op<"convolution", [Pure]> {
  let summary = "Convolution operation";
  let description = [{
    Computes dot products between windows of `lhs` and slices of `rhs` and
    produces `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

    Example:
    ```mlir
    %result = "mhlo.convolution"(%lhs, %rhs) {
      window_strides = dense<4> : tensor<2xi64>,
      padding = dense<0> : tensor<2x2xi64>,
      lhs_dilation = dense<2> : tensor<2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_reversal = dense<false> : tensor<2xi1>,
      dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x4x4x1xi32>, tensor<3x3x1x1xi32>) -> tensor<1x2x2x1xi32>
    ```
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

def MHLO_CopyOp: MHLO_Op<"copy",
      [Pure, Elementwise, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Copy operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation a copy of `operand`. Depending on the metadata
    attached to the operation, it can behave quite differently from a no-op.

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
  let summary = "CrossReplicaSum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AllReduceOp with
    `channel_id = 0`, `use_global_device_ids = false` and `computation`
    implementing addition:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#all_reduce

    Example:
    ```mlir
    %result = "mhlo.cross-replica-sum"(%operand) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs MHLO_Tensor);
}

def MHLO_CustomCallOp: MHLO_Op<"custom_call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "CustomCall operation";
  let description = [{
    Encapsulates an implementation-defined operation `call_target_name` that
    takes `inputs` and `called_computations` and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#custom_call

    Example:
    ```mlir
    %results = "mhlo.custom_call"(%input0) {
      call_target_name = "foo",
      has_side_effect = false,
      backend_config = "bar",
      api_version = 1 : i32,
      called_computations = [@foo]
    } : (tensor<f32>) -> tensor<f32>

    A custom call invokes code external to XLA. The `inputs` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    If XLA runtime is enabled for a backend, then custom calls use the runtime
    custom call calling convention to call into the external functions. This
    calling convention defines an ABI for encoding arguments, attributes and
    results.

    Depending on the API version there are two ways to pass extra bits of static
    information to the external function:

    1. For `API_VERSION_TYPED_FFI` custom calls `backend_config` must be a
       dictionary attribute, that will be encoded according to the custom call
       calling convention and passed to the external function as the attributes
       argument. External code is expected to use declarative bindings (see
       `xla/runtime/custom_call.h`) to decode them at run time. These custom
       calls are only supported if XLA uses XLA runtime.

    2. For previous API versions it is the user responsibility to encode extra
       bits of static information as a string `backend_config` attribute, and
       decode it at run time.
    ```
  }];
  let arguments = (ins
    Variadic<MHLO_CustomCallValue>:$inputs,
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
  let results = (outs Variadic<MHLO_CustomCallValue>);
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
  let summary = "Dot operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as XLA's Dot:
    https://www.tensorflow.org/xla/operation_semantics#dot

    Example:
    ```mlir
    %0 = mhlo.dot %arg0, %arg1 : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<1x1xi32>
    ```
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
  let summary = "DotGeneral operation";
  let description = [{
    Computes dot products between slices of `lhs` and slices of `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general

    Example:
    ```mlir
    %result = "mhlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [1]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x2x2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs MHLO_Tensor);
  // DotGeneral op required custom exporter to pass the preferred element type
  // to Xla builder.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_SparseDotOp: MHLO_Op<"sparse_dot", [Pure]> {
  let summary = "Sparse dot operation";
  let description = [{
    Similar to `dot_general` operation, with one or both of the operands being
    sparse. An additional argument provides sparsity meta information.
    Disclaimer: this op is experimental / a work in progress.
  }];
  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    Variadic<MHLO_Tensor>:$meta,
    OptionalAttr<MHLO_SparsityDescriptor>:$lhs_sparsity,
    OptionalAttr<MHLO_SparsityDescriptor>:$rhs_sparsity,
    MHLO_DotDimensionNumbers:$dot_dimension_numbers,
    MHLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs MHLO_Tensor);
  // SparseDot op required custom exporter to pass the preferred element type
  // to Xla builder.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def MHLO_EinsumOp: MHLO_Op<"einsum", [Pure]> {
  let summary = "Einsum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as TF's einsum:
    https://www.tensorflow.org/api_docs/python/tf/einsum

    Example:
    ```mlir
    %result = "mhlo.einsum"(%lhs, %rhs) {
      einsum_config = "ab,bc->ac"
    } : (tensor<4x16xf32>, tensor<16x4xf32>) -> tensor<4x4xf32>
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$lhs,
    MHLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );

  let results = (outs MHLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def MHLO_UnaryEinsumOp: MHLO_Op<"unary_einsum", [Pure]> {
  let summary = "UnaryEinsum operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as TF's einsum:
    https://www.tensorflow.org/api_docs/python/tf/einsum

    Example:
    ```mlir
    %result = "mhlo.unary_einsum"(%operand) {
      einsum_config = "ab->a"
    } : (tensor<4x16xf32>) -> tensor<4xf32>
    ```
  }];

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
  let summary = "Fft operation";
  let description = [{
    Performs the forward and inverse Fourier transforms for real and complex
    inputs/outputs.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#fft

    Example:
    ```mlir
    %result = mhlo.fft %operand, type = FFT, length = [4] : (tensor<4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    MHLO_FftTypeAttr:$fft_type,
    I64ElementsAttr:$fft_length
  );

  let results = (outs MHLO_Tensor);
}

def MHLO_GatherOp: MHLO_Op<"gather", [InferTensorTypeWithReify, Pure]> {
  let summary = "Gather operation";
  let description = [{
    Gathers slices from `operand` tensor from offsets specified in
    `start_indices` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "mhlo.gather"(%operand, %start_indices) {
      dimension_numbers = #mhlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [0, 2],
        index_vector_dim = 2>,
      slice_sizes = dense<[0, 2, 2]> : tensor<3xi64>,
      indices_are_sorted = false
    } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
    ```
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
  let summary = "GetDimensionSize operation";
  let description = [{
    Produces the size of the given `dimension` of the `operand`.

    See
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

    Example:
    ```mlir
    %result = mhlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>
    ```
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
}

def MHLO_MapOp: MHLO_ShapedInterfaceOp<"map",
      [RecursiveMemoryEffects, SameOperandsAndResultShape,
       SingleBlockImplicitTerminator<"ReturnOp">, InferTensorTypeWithReify]> {
  let summary = "Map operation";
  let description = [{
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "mhlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = mhlo.multiply %arg0, %arg1 : tensor<i32>
        mhlo.return %0 : tensor<i32>
    }) {
      dimensions = dense<[0, 1]> : tensor<2xi64>
    } : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    ```
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
      [Pure, HLO_CompatibleOperandsAndResultElementType]> {
  let summary = "Reshape operation";
  let description = [{
    Performs reshape of `operand` tensor to a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = mhlo.reshape %operand : (tensor<2xf32>) -> tensor<1x2xf32>
    ```
  }];

  let arguments = (ins MHLO_AnyTensor:$operand);

  let results = (outs MHLO_StaticShapeTensor);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_DynamicReshapeOp: MHLO_ShapedInterfaceOp<"dynamic_reshape", [Pure]> {
  let summary = "DynamicReshape operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ReshapeOp except that the
    result shape is specified dynamically via `output_shape`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %0 = mhlo.dynamic_reshape %arg0, %shape : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
    ```
  }];

  let arguments = (ins MHLO_AnyTensor:$operand, MHLO_DimensionTensor:$output_shape);
  let results = (outs MHLO_AnyTensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_ScatterOp: MHLO_Op<"scatter",
      [SameVariadicOperandSize, RecursiveMemoryEffects,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Scatter operation";
  let description = [{
    Produces `results` tensors which are equal to `inputs` tensors except that
    several slices specified by `scatter_indices` are updated with the values
    `updates` using `update_computation`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

   Example:
   ```mlir
   %result = "mhlo.scatter"(%input, %scatter_indices, %update) ({
     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
       %0 = mhlo.add %arg0, %arg1 : tensor<i32>
       mhlo.return %0 : tensor<i32>
   }) {
     scatter_dimension_numbers = #mhlo.scatter<
       update_window_dims = [2,3],
       inserted_window_dims = [0],
       scatter_dims_to_operand_dims = [1, 0],
       index_vector_dim = 2>,
     indices_are_sorted = false,
     unique_indices = false
   } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<2x3x2x2xi32>) -> tensor<3x4x2xi32>
   ```
  }];
  let arguments = (ins
    Variadic<MHLO_Tensor>:$inputs,
    RankedTensorOf<[AnyInteger, Index]>:$scatter_indices,
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
  let summary = "Select operation";
  let description = [{
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = mhlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
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
  let summary = "SelectAndScatter operation";
  let description = [{
    Scatters the values from the `source` tensor using `scatter` based on the
    outcome of `reduce_window` of the `input` tensor using `select` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select_and_scatter

    Example:
    ```mlir
    %result = "mhlo.select_and_scatter"(%operand, %source, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction GE>
        } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      window_dimensions = dense<[3, 1]> : tensor<2xi64>,
      window_strides = dense<[2, 1]> : tensor<2xi64>,
      padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<4x2xi32>, tensor<2x2xi32>, tensor<i32>) -> tensor<4x2xi32>
    ```
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
      InferTensorType]> {
  let summary = "SetDimensionSize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as XLA's SetDimensionSize:
    https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

    Example:
    ```mlir
    %0 = mhlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs MHLO_Tensor);

  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

def MHLO_SortOp : MHLO_Op<"sort",
      [RecursiveMemoryEffects, SameOperandsAndResultShape, InferTensorType]> {
  let summary = "Sort operation";
  let description = [{
    Sorts a variadic number of tensors in `inputs` together, according to a
    custom `comparator`, along the given `dimension` and produces a variadic
    number of tensors as `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort

    Example:
    ```mlir
    %result0, %result1 = "mhlo.sort"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
        %predicate = "mhlo.compare"(%arg0, %arg1) {
          comparison_direction = #stablehlo<comparison_direction GT>
          } : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "mhlo.return"(%predicate) : (tensor<i1>) -> ()
    }) {
      dimension = 0 : i64,
      is_stable = true
    } : (tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>)
    ```
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

def MHLO_TopKOp : MHLO_Op<"topk", [RecursiveMemoryEffects, InferTensorType]> {
  let summary = "TopK operation";
  let description = [{
    Returns top `k` values and their indices, along the last
    dimension of the operand if `largest=true` or the bottom `k` values if
    `largest=false`.

    See:
    https://www.tensorflow.org/xla/operation_semantics#top-k

    Example:
    ```mlir
    %values, %indices = mhlo.topk(%operand, k=5, largest=true)
      : tensor<100xf32> -> (tensor<5xf32>, tensor<5xi32>)
    ```
  }];

  let arguments = (ins
    MHLO_Tensor:$operand,
    I64Attr:$k,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$largest
  );
  let results = (outs MHLO_Tensor:$values,
                      MHLO_Tensor:$indices);

  let assemblyFormat = [{
    `(`$operand `,` `k` `=` $k (`,` `largest` `=` $largest^)? `)` attr-dict `:`
    type($operand) `->` `(`type($values)`,` type($indices)`)`
  }];
  let hasCustomHLOConverter = 1;
}

def MHLO_ReverseOp: MHLO_Op<"reverse",
      [Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Reverse operation";
  let description = [{
    Reverses the order of elements in the `operand` along the specified
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse

    Example:
    ```mlir
    %result = mhlo.reverse %operand, dims = [1] : tensor<3x2xi32>
    ```
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
  let summary = "PartitionId operation";
  let description = [{
    Produces `partition_id` of the current process.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#partition_id

    Example:
    ```mlir
    %result = mhlo.partition_id : tensor<ui32>
    ```
  }];
  let results = (outs UI32RankedTensor);
  let results = (outs UI32RankedTensor);
  let hasCustomHLOConverter = 1;

  let assemblyFormat = "attr-dict `:` type(results)";
}

def MHLO_PadOp: MHLO_ShapedInterfaceOp<"pad",
      [Pure, SameOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Pad operation";
  let description = [{
    Expands `operand` by padding around the tensor as well as between the
    elements of the tensor with the given `padding_value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %0 = mhlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1], interior = [1, 2]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
    ```
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
  let summary = "Trace operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/604.

    It is not used by JAX, PyTorch or TensorFlow, so it looks like we should've
    classified it as "Private to XLA" and not included it in StableHLO in the
    first place. With that in mind, its semantics will not be documented here.

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
      [Pure, HLO_CompatibleOperandsAndResultElementType,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose

    Example:
    ```mlir
    %0 = mhlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x2x3xi32>) -> tensor<3x2x1xi32>
    ```
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
  let summary = "TriangularSolve operation";
  let description = [{
    Solves batches of systems of linear equations with lower or upper triangular
    coefficient matrices.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#triangular_solve

    Example:
    ```mlir
    %result = "mhlo.triangular_solve"(%a, %b) {
      left_side = true,
      lower = true,
      unit_diagonal = false,
      transpose_a = #stablehlo<transpose NO_TRANSPOSE>
    } : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
    ```
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
  let summary = "ReduceWindow operation";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values`
    and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

    Example:
    ```mlir
    %result = "mhlo.reduce_window"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = mhlo.add %arg0, %arg1 : tensor<i32>
        mhlo.return %0 : tensor<i32>
    }) {
      window_dimensions = dense<[2, 1]> : tensor<2xi64>,
      window_strides = dense<[4, 1]> : tensor<2xi64>,
      base_dilations = dense<[2, 1]> : tensor<2xi64>,
      window_dilations = dense<[3, 1]> : tensor<2xi64>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<3x2xi32>, tensor<i32>) -> tensor<2x2xi32>
    ```
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
  let summary = "Return operation";
  let summary = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/425.

    Informally, this operation serves as a terminator for regions defined by
    the StableHLO ops. Non-StableHLO ops, e.g. `func.func`, have their own
    terminators, e.g. `func.return`.

    Example:
    ```mlir
    %result = "mhlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "mhlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = dense<1> : tensor<1xi64>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
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
  let summary = "TorchIndexSelect operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as PyTorch's index_select,
    augmented with support for batch dimensions:
    https://pytorch.org/docs/stable/generated/torch.index_select.html.

    The `batch_dims` attribute specifies the number of major batch dimensions
    (0 or more) that act like a multidimensional loop over both the operand and
    the index.

    Example:
    ```mlir
    %result = "mhlo.torch_index_select"(%operand, %index) {
      dim = 2 : i64,
      batch_dims = 1 : i64
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
  let summary = "OptimizationBarrier operation";
  let description = [{
    Ensures that the operations that produce the `operand` are executed before any
    operations that depend on the `result` and prevents compiler transformations
    from moving operations across the barrier. Other than that, the operation is
    an identity, i.e. `result` = `operand`.

    See
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#optimization_barrier

    Example:
    ```mlir
    %result0, %result1 = mhlo.optimization_barrier %operand0, %operand1 : tensor<f32>, tensor<f32>
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
  let summary = "Rng operation";
  let description = [{
    Generates random numbers using the `rng_distribution` algorithm and produces
    a `result` tensor of a given shape `shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng

    Example:
    ```mlir
    %result = mhlo.rng %a, %b, %shape, distribution = NORMAL : (tensor<i32>, tensor<i32>, tensor<2xi64>) -> tensor<3x3xi32>
    ```
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
  let summary = "RngBitGenerator operation";
  let description = [{
    Returns an `output` filled with uniform random data and an updated output
    state `output_state` given an initial state `initial_state` using the
    pseudorandom number generator algorithm `rng_algorithm`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator

    Example:
    ```mlir
    %output_state, %output = mhlo.rng_bit_generator %initial_state, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)
    ```
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
  let summary = "XlaRngGetAndUpdateState operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation represents the change of the global random number
    generator state for rng instructions. The global state is incremented by
    delta and the old state is returned.

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
      [Pure], RankedTensorOf<[MHLO_Float, MHLO_QuantizedInt]>,
      MHLO_QuantizedIntTensor> {
  let summary = "UniformQuantize operation";
  let description = [{
    Performs element-wise conversion of floating-point tensor or quantized
    tensor `operand` to a quantized tensor `result` according to the
    quantization parameters defined by the `result` type.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_quantize

    Example:
    ```mlir
    %result = mhlo.uniform_quantize %operand : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
    ```
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def MHLO_UniformDequantizeOp : MHLO_UnaryElementwiseOp<"uniform_dequantize",
      [InferTensorType, Pure], MHLO_QuantizedIntTensor, MHLO_FpTensor> {
  let summary = "UniformDequantize operation";
  let description = [{
    Performs element-wise conversion of quantized tensor `operand` to a
    floating-point tensor `result` according to the quantization parameters
    defined by the `operand` type.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#uniform_dequantize

    Example:
    ```mlir
    %result = mhlo.uniform_dequantize %operand : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
    ```
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def MHLO_FusionOp : MHLO_Op<"fusion", []> {
  let summary = "Fusion operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation consists of a group of basic ops (represented as
    a region attached to it). It serves as a hint to the backend that it is
    beneficial to emit the contained ops into a single loop nest or kernel.
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
  let summary = "Bitcast operation";
  let description = [{
    This operation is private to the XLA compiler, so it is does not yet have
    a specification.

    Informally, this operation changes the shape of the input in the way that
    the physical arrangement of elements are unchanged.

    This operation needs layout information to make sense of "physical
    arrangement of elements", and layout support in MHLO is currently a work
    in progress.

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

def MHLO_ReducePrecisionOp : MHLO_Op<"reduce_precision",
      [HLO_CompatibleOperandsAndResultType, Pure, Elementwise]> {
  let summary = "ReducePrecision operation";
  let description = [{
    Performs element-wise conversion of `operand` to another floating-point type
    that uses `exponent_bits` and `mantissa_bits` and back to the original
    floating-point type and produces an `output` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_precision

    Example:
    ```mlir
    %output = mhlo.reduce_precision %operand, format = e5m2 : tensor<6xf32>
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
  let summary = "RealDynamicSlice operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as SliceOp except
    that `start_indices`, `limit_indices` and `strides` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = mhlo.real_dynamic_slice %operand,
                %start_indices, %limit_indices, %strides
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
  let summary = "DynamicPad operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as PadOp except
    that `edge_padding_low`, `edge_padding_high` and `interior_padding` are
    specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %result = mhlo.dynamic_pad %operand, %padding_value,
                %edge_padding_low, %edge_padding_high, %interior_padding
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
  let summary = "DynamicGather operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as GatherOp except
    that `slice_sizes` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather

    Example:
    ```mlir
    %result = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
      dimension_numbers = #mhlo.gather<
        offset_dims = [2, 3],
        collapsed_slice_dims = [0],
        start_index_map = [0, 2],
        index_vector_dim = 2>,
      indices_are_sorted = false
    } : (tensor<3x4x2xi32>, tensor<2x3x2xi64>, tensor<3xi64>) -> tensor<2x3x2x2xi32>
    ```
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
  let summary = "DynamicConv operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ConvolutionOp except
    that `padding` is specified dynamically via `d_padding`:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convolution

    Example:
    ```mlir
    %result = "mhlo.dynamic_conv"(%lhs, %rhs, %d_padding) {
      window_strides = dense<4> : tensor<2xi64>,
      lhs_dilation = dense<2> : tensor<2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_reversal = dense<false> : tensor<2xi1>,
      dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
      feature_group_count = 1 : i64,
      batch_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x4x4x1xi32>, tensor<3x3x1x1xi32>, tensor<2x2xi64>) -> tensor<1x2x2x1xi32>
    ```
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
  let summary = "ComputeReshapeShape operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation computes an output_shape for DynamicReshapeOp
    from the `num_elements` number of elements in an operand of DynamicReshapeOp
    and the `dynamic_shape` shape provided to TF's reshape:
    https://www.tensorflow.org/api_docs/python/tf/reshape

    For example, for `num_elements = 12` and `dynamic_shape = [2, -1]`,
    the `result` is going to be `[2, 6]`. If operands are not valid (e.g. if
    dimensions do not evenly divide the number of elements, or if there are
    multiple -1 values in dimensions), this leads to undefined behavior.

    Example:
    ```mlir
    %result = mhlo.compute_reshape_shape %num_elements, %dynamic_shape
           : (index, tensor<2xi32>) -> tensor<2xi32>
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def MHLO_CstrReshapableOp :
    MHLO_Op<"cstr_reshapable", [Pure]> {
  let summary = "CstrReshapable operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation creates a witness on the constraint that
    ComputeReshapeShape would succeed with the provided operands.

    Example:
    ```mlir
    %result = mhlo.cstr_reshapable %num_elements, %dynamic_shape
           : (index, tensor<3xi32>) -> !shape.witness
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs Shape_WitnessType:$result);

  let hasCustomHLOConverter = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS
