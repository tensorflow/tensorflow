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

#ifndef HLO_OPS
#define HLO_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_utils.td"

class HLO_Op<string mnemonic, list<Trait> traits> :
    Op<HLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;
}

class HLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    HLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
}

//===----------------------------------------------------------------------===//
// MHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def HLO_ConstOp : HLO_Op<"constant",
    [ConstantLike, NoSideEffect, AllTypesMatch<["value", "output"]>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    HLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  let assemblyFormat = "attr-dict $value";

  let hasFolder = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_IotaOp : HLO_Op<"iota", [NoSideEffect]> {
  let summary = "Iota operator";
  let description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.
  }];
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs HLO_IntFpOrComplexTensor:$output);

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_DynamicIotaOp: HLO_ShapedInterfaceOp<"dynamic_iota", [NoSideEffect]> {
  let summary = "Create linear increasing values from 0 to length -1.";
  let description = [{
    Produces an HLO Tensor of the specified shape, with an incremental set of
    values along the specified dimension starting at 0.

    Requires:
    - The output length of the tensor result.
  }];

  let arguments = (ins HLO_DimensionTensor:$output_shape, I64Attr:$iota_dimension);
  let results = (outs HLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
}


def HLO_CreateTokenOp : HLO_Op<"create_token", [NoSideEffect]> {
  let summary = "Create Token operator";

  let description = [{
    Produces a HLO token. Tokens are used for ordering side-effecting perations.
    This is exported to HLO as an AfterAll operation with no operands to
    generate a token.
  }];

  let results = (outs HLO_Token:$output);
}

//===----------------------------------------------------------------------===//
// MHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class HLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : HLO_Op<mnemonic, traits # [Elementwise,
    InferShapedTypeOpInterface, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::mhlo::deriveShapeFromOperand(&builder, getOperation(),
                                                  operands.front(),
                                                  &reifiedReturnShapes);
    }
    // Replace the strict default implementation with one that allows
    // for different sparsity encodings during the is-same test.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size() != r.size()) return false;
      for (auto it : llvm::zip(l, r))
        if (!isSameTypesWithoutSparseEncoding(std::get<0>(it), std::get<1>(it)))
          return false;
      return true;
    }
  }];
  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return ::mlir::mhlo::parseUnaryOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      ::mlir::mhlo::printUnaryOp(getOperation(), p);
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

// Abs supports complex to real, so element type is not guaranteed to match.
def HLO_AbsOp: HLO_UnaryElementwiseOp<"abs",
    [NoSideEffect,
     DeclareOpInterfaceMethods<InferTypeOpInterface>],
     TensorOf<[HLO_SInt, AnyFloat, HLO_Complex]>> {
  let summary = "Absolute value operator";
  let description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def HLO_CbrtOp: HLO_UnaryElementwiseOp<"cbrt",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor> {
  let summary = "Cubic root operator";
  let description = [{
    Returns element-wise cubic root of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_CeilOp: HLO_UnaryElementwiseOp<"ceil",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor> {
  let summary = "Ceil operator";
  let description = [{
    Returns `Ceil(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_ConvertOp : HLO_UnaryElementwiseOp<"convert",
    [NoSideEffect, SameOperandsAndResultShape], HLO_Tensor> {
  let summary = "Convert operator";
  let description = [{
    Performs element-wise conversion of values from one type to another, e.g.
    float to int.

    See https://www.tensorflow.org/xla/operation_semantics#convertelementtype.
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];

  let hasFolder = 1;

  let hasCanonicalizer = 1;

  let hasCustomHLOConverter = 1;
}

def HLO_ClzOp: HLO_UnaryElementwiseOp<"count_leading_zeros",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntTensor> {
  let summary = "Count-leading-zeros (Clz) operator";
  let description = [{
    Returns the number of leading zeros in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_CosOp: HLO_UnaryElementwiseOp<"cosine",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Cos operator";
  let description = [{
    Returns `Cos(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_ExpOp: HLO_UnaryElementwiseOp<"exponential",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Exponential operator";
  let description = [{
    Returns `e^(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_Expm1Op: HLO_UnaryElementwiseOp<"exponential_minus_one",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Exponential minus one operator";
  let description = [{
    Returns `e^(operand) - 1` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_FloorOp: HLO_UnaryElementwiseOp<"floor",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor> {
  let summary = "Floor operator";
  let description = [{
    Returns `Floor(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_ImagOp: HLO_UnaryElementwiseOp<"imag",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    HLO_FpOrComplexTensor> {
  let summary = "Imag operator";
  let description = [{
    Returns `Imag(operand)` element-wise.
  }];
  let results = (outs HLO_FpTensor);
  let hasFolder = 1;
}

def HLO_IsFiniteOp: HLO_UnaryElementwiseOp<"is_finite", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>], HLO_Tensor> {
  let summary = "IsFinite operator";
  let description = [{
    Tests whether each element of operand is finite, i.e., is not positive or
    negative infinity, and is not NaN. Returns a tensor of 1-bit integers with
    the same shape as the input, where each element is nonzero (i.e. true) if
    and only if the corresponding input element is finite.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let arguments = (ins HLO_FpTensor:$x);
  let results = (outs HLO_PredTensor:$y);
}

def HLO_LogOp: HLO_UnaryElementwiseOp<"log",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Logarithm operator";
  let description = [{
    Returns `log(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_Log1pOp: HLO_UnaryElementwiseOp<"log_plus_one",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Log1p operator";
  let description = [{
    Returns `log(operand+1)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_LogisticOp: HLO_UnaryElementwiseOp<"logistic",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Logistic operator";
  let description = [{
    Returns `logistic(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_NotOp: HLO_UnaryElementwiseOp<"not",
    [NoSideEffect, SameOperandsAndResultType], HLO_PredOrIntTensor> {
  let summary = "Not operator";
  let description = [{
    Returns `!operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
    let hasFolder = 1;
}

def HLO_NegOp: HLO_UnaryElementwiseOp<"negate",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntFpOrComplexTensor> {
  let summary = "Negation operator";
  let description = [{
    Returns `-operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let hasFolder = 1;
}

def HLO_PopulationCountOp: HLO_UnaryElementwiseOp<"popcnt",
    [NoSideEffect, SameOperandsAndResultType], HLO_IntTensor> {
  let summary = "PopulationCount operator";
  let description = [{
    Returns the number of bits set in each operand element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_RealOp: HLO_UnaryElementwiseOp<"real",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>],
    HLO_FpOrComplexTensor> {
  let summary = "Real operator";
  let description = [{
    Returns `Real(operand)` element-wise.
  }];
  let results = (outs HLO_FpTensor);
  let hasFolder = 1;
}

def HLO_RoundOp: HLO_UnaryElementwiseOp<"round_nearest_afz",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpTensor> {
  let summary = "Round operator";
  let description = [{
    Returns `Round(operand)` element-wise, rounding to nearest integer with
    half-way cases rounding away from zero.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let hasFolder = 1;
}

def HLO_RsqrtOp: HLO_UnaryElementwiseOp<"rsqrt",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Reciprocal Square-root operator";
  let description = [{
    Returns `1.0 / sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_SignOp: HLO_UnaryElementwiseOp<"sign",
    [NoSideEffect, SameOperandsAndResultType],
    TensorOf<[HLO_SInt, AnyFloat, HLO_Complex]>> {
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
  }];
  let hasFolder = 1;
}
def HLO_SinOp: HLO_UnaryElementwiseOp<"sine",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Sin operator";
  let description = [{
    Returns `Sin(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
def HLO_SqrtOp: HLO_UnaryElementwiseOp<"sqrt",
    [NoSideEffect, SameOperandsAndResultType], HLO_FpOrComplexTensor> {
  let summary = "Square-root operator";
  let description = [{
    Returns `sqrt(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
  let hasFolder = 1;
}

def HLO_TanhOp: HLO_UnaryElementwiseOp<"tanh",
    [NoSideEffect, SameOperandsAndResultType],
    HLO_FpOrComplexTensor> {
  let summary = "Tanh operator";
  let description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}
//===----------------------------------------------------------------------===//
// MHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class HLO_BinaryElementwiseOpNoAssembly<string mnemonic, list<Trait> traits> :
    HLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs
  );

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::mhlo::deriveShapeFromOperand(&builder, getOperation(),
                                                  operands.front(),
                                                  &reifiedReturnShapes);
    }
    // Replace the strict default implementation with one that allows
    // for different sparsity encodings during the is-same test.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      // TODO(burmako): Re-enable this check by accounting for dynamism rules.
      // if (l.size() != r.size()) return false;
      // for (auto it : llvm::zip(l, r))
      //   if (!isSameTypesWithoutSparseEncoding(std::get<0>(it), std::get<1>(it)))
      //    return false;
      return true;
    }
  }];

  let results = (outs HLO_Tensor:$result);
}

class HLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits> :
    HLO_BinaryElementwiseOpNoAssembly<mnemonic, traits> {
  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      return ::mlir::mhlo::parseBinaryOp(parser, result);
    }
    void $cppClass::print(OpAsmPrinter &p) {
      ::mlir::mhlo::printBinaryOp(getOperation(), p);
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def HLO_AddOp : HLO_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, InferTensorType]> {
  let summary = "Addition operator";
  let description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_Atan2Op : HLO_BinaryElementwiseOp<"atan2",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Atan2 operator";
  let description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_ComplexOp: HLO_BinaryElementwiseOpNoAssembly<"complex",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Complex operator";
  let description = [{
    Performs element-wise conversion of a pair of real and imaginary values to
    a complex value.
  }];
  let arguments = (ins HLO_FpTensor:$lhs, HLO_FpTensor:$rhs);
  let results = (outs HLO_ComplexTensor:$result);
  let assemblyFormat = "`(`operands`)` attr-dict `:` `(`type(operands)`)` `->` type($result)";
  let hasFolder = 1;
}

def HLO_DivOp : HLO_BinaryElementwiseOp<"divide",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Division operator";
  let description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_MaxOp : HLO_BinaryElementwiseOp<"maximum",
      [Commutative, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Maximum operator";
  let description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_MinOp : HLO_BinaryElementwiseOp<"minimum",
      [Commutative, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Minimum operator";
  let description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_MulOp : HLO_BinaryElementwiseOp<"multiply",
      [Commutative, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Multiplication operator";
  let description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_PowOp : HLO_BinaryElementwiseOp<"power",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Power operator";
  let description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}
def HLO_RemOp : HLO_BinaryElementwiseOp<"remainder",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Remainder operator";
  let description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

def HLO_ShiftLeftOp : HLO_BinaryElementwiseOp<"shift_left",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Shift Left operator";
  let description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_ShiftRightArithmeticOp : HLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Shift right arithmetic operator";
  let description = [{
    Returns arithmetic `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_ShiftRightLogicalOp : HLO_BinaryElementwiseOp<"shift_right_logical",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Shift right logical operator";
  let description = [{
    Returns logical `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_SubOp : HLO_BinaryElementwiseOp<"subtract",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Subtraction operator";
  let description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MHLO binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class HLO_BinaryLogicalElementwiseOp<string mnemonic> :
        HLO_BinaryElementwiseOp<
            mnemonic, [Commutative, NoSideEffect, SameOperandsAndResultType]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs
  );

  let hasFolder = 1;
}

def HLO_AndOp: HLO_BinaryLogicalElementwiseOp<"and"> {
  let summary = "Logical and";
  let description = [{
    Returns `logical_and(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_OrOp: HLO_BinaryLogicalElementwiseOp<"or"> {
  let summary = "Logical or";
  let description = [{
    Returns `logical_or(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLO_XorOp : HLO_BinaryLogicalElementwiseOp<"xor"> {
  let summary = "Logical xor";
  let description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// MHLO communication op definitions.
//===----------------------------------------------------------------------===//

// InfeedOp corresponds to 'InfeedWithToken' xla client API and not 'Infeed'.
// InfeedWithToken allows ordering of infeed HLO instructions using tokens.
def HLO_InfeedOp : HLO_Op<"infeed", []> {

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
    HLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$infeed_config,
    OptionalAttr<ArrayAttr>:$layout
  );
  let results = (outs Variadic<HLO_TensorOrToken>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

// OutfeedOp corresponds to 'OutfeedWithToken' xla client API and not 'Outfeed'.
// OutfeedWithToken allows ordering of outfeed HLO instructions using tokens.
def HLO_OutfeedOp : HLO_Op<"outfeed", []> {

  let summary = "Outfeed operator";

  let description = [{
    Generates outgoing data transfers for the given data. It takes data and a
    token type operand and produces a token type value. Tokens are used for
    ordering side-effecting operations.

    See https://www.tensorflow.org/xla/operation_semantics#outfeed.
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$operands,
    HLO_Token:$token,
    DefaultValuedStrAttr<StrAttr, "">:$outfeed_config
  );
  let results = (outs HLO_Token);
  let hasCustomHLOConverter = 1;
}

def HLO_SendOp : HLO_Op<"send", []> {

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
    Variadic<HLO_Tensor>:$operands,
    HLO_Token:$token,
    ChannelHandle:$channel_handle,
    DefaultValuedAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs HLO_Token);
  let hasCustomHLOConverter = 1;
}

def HLO_RecvOp : HLO_Op<"recv", []> {

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
    HLO_Token:$token,
    ChannelHandle:$channel_handle,
    DefaultValuedAttr<BoolAttr, "false">:$is_host_transfer
  );

  let results = (outs Variadic<HLO_TensorOrToken>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MHLO parallelism related op definitions.
//===----------------------------------------------------------------------===//

def HLO_ReplicaIdOp : HLO_Op<"replica_id", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ReplicaId operator";
  let description = [{
    Returns the unique ID (int32 scalar) of the replica.

    The unique ID of each replica is an unsigned integer in the interval [0, N),
    where N is the number of replicas. Since all the replicas are running the
    same program, a ReplicaId() call in the program will return a different
    value on each replica.

    See https://www.tensorflow.org/xla/operation_semantics#replicaid.
  }];
  let results = (outs TensorOf<[UI32]>);
}

//===----------------------------------------------------------------------===//
// MHLO control flow op definitions.
//===----------------------------------------------------------------------===//

def HLO_AddDependencyOp : HLO_Op<"add_dependency", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AddDependency operator";
  let description = [{
    AddDependency takes two operands: a data operand and a token. The output of
    the operation is the data operand. When used with AfterAll this operation
    enables ordering non-side-effecting operations (those that do not produce
    token values).
  }];

  let arguments = (ins HLO_TensorOrToken:$operand, HLO_Token:$token);
  let results = (outs HLO_TensorOrToken:$output);
  let hasCustomHLOConverter = 1;
}


def HLO_AfterAllOp : HLO_Op<"after_all", [NoSideEffect]> {

  let summary = "AfterAll operator";

  let description = [{
    AfterAll takes a variadic number of tokens and produces a single token.
    Tokens are primitive types which can be threaded between side-effecting
    operations to enforce ordering. AfterAll can be used as a join of tokens
    for ordering a operation after a set operations.

    See https://www.tensorflow.org/xla/operation_semantics#afterall.
  }];

  let arguments = (ins Variadic<HLO_Token>:$operands);
  let results = (outs HLO_Token);
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. IfOp maps to predicated
// conditional use of kConditional HLO.
def HLO_IfOp: HLO_Op<"if", [
    RecursiveSideEffects,
    SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "If operator";

  let description = [{
    Returns the result of executing either a true or false function depending on
    the result of a condition function.

    See https://www.tensorflow.org/xla/operation_semantics#conditional.
  }];

  let arguments = (ins
    HLO_PredTensor:$pred
  );

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<HLO_TensorOrToken>);

  // TODO(b/129422361): ConditionalOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

// Xla Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def HLO_CaseOp: HLO_Op<"case", [
      RecursiveSideEffects,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> {
  let summary = "Switch-Case operator";
  let description = [{
    Returns the result of executing `branches[index]`. If
    `index` is < 0 or >= N, then `branches[N-1] is executed as
    the default branch.

    Each branch `branches[b]` must take in a single argument of same type as
    `branch_operands[b]` and will be invoked with `branch_operands[b]`. The type
    of the returned value of each branch must be the same.

    Note that only one of the branches will be executed depending on the value
    of index.
    See https://www.tensorflow.org/xla/operation_semantics#conditional.
  }];

  let arguments = (ins
    I32Tensor:$index
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let results = (outs Variadic<HLO_TensorOrToken>);

  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}


def HLO_WhileOp: HLO_Op<"while", [
      RecursiveSideEffects,
      HLO_PairwiseSameOperandAndResultType,
      SingleBlockImplicitTerminator<"ReturnOp">,
      OpAsmOpInterface
    ]> {
  let summary = "While operator";
  let description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];
  let arguments = (ins Variadic<HLO_TensorOrToken>:$operand);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let results = (outs Variadic<HLO_TensorOrToken>);

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
  }];
  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HLO_AllGatherOp : HLO_Op<"all_gather", [SameOperandsAndResultElementType]> {

  string summary = "AllGather operator";

  string description = [{
    Performs concatenation across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allgather
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$all_gather_dim,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<ChannelHandle>:$channel_handle
  );
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_AllReduceOp : HLO_Op<"all_reduce",
    [SameOperandsAndResultType]> {
  let summary = "AllReduce operator";
  let description = [{
    Performs a custom reduction across replicas.

    See https://www.tensorflow.org/xla/operation_semantics#allreduce.
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<ChannelHandle>:$channel_handle
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;
}

def HLO_ReduceScatterOp : HLO_Op<"reduce_scatter",
    [SameOperandsAndResultElementType]> {
  let summary = "ReduceScatter operator";
  let description = [{
     Performs all_reduce followed by a scatter.

     See https://www.tensorflow.org/xla/operation_semantics#reducescatter
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$scatter_dimension,
    I64ElementsAttr:$replica_groups,
    OptionalAttr<ChannelHandle>:$channel_handle
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_AllToAllOp : HLO_Op<"all_to_all",
    [NoSideEffect, SameOperandsElementType, SameOperandsShape]> {

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$split_dimension,
    I64Attr:$concat_dimension,
    I64Attr:$split_count,
    I64ElementsAttr:$replica_groups
  );
  let results = (outs HLO_Tensor);

  let hasVerifier = 1;
}

def HLO_ReduceOp: HLO_ShapedInterfaceOp<"reduce", [
      RecursiveSideEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> {
  let summary = "Reduce operator";
  let description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$operands,
    Variadic<HLO_Tensor>:$init_values,
    I64ElementsAttr:$dimensions
  );

  let results = (outs Variadic<HLO_Tensor>);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ValueRange":$init_values,
      "DenseIntElementsAttr":$dimensions)>];

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
def HLO_GetTupleElementOp: HLO_Op<"get_tuple_element", [NoSideEffect,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "GetTupleElement operator";
  let description = [{
    Returns a member of a tuple specified by an index.

    See https://www.tensorflow.org/xla/operation_semantics#gettupleelement.
  }];
  let arguments = (ins
    HLO_Tuple,
    I32Attr:$index
  );

  let results = (outs HLO_TensorOrTokenOrTuple);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_TupleOp : HLO_Op<"tuple", [NoSideEffect,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "XLA's tuple op";
  let description = [{
     Groups a set of tensor inputs into a single tuple object.

     See https://www.tensorflow.org/xla/operation_semantics#tuple.
   }];
  let arguments = (ins Variadic<HLO_TensorOrTokenOrTuple>:$val);
  let results = (outs HLO_Tuple);

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def HLO_CompareOp: HLO_Op<"compare", [NoSideEffect, SameOperandsElementType,
    SameOperandsAndResultShape, Elementwise, InferTensorTypeWithReify]> {
  let summary = "Comparison operator";
  let description = [{
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
    HLO_ComparisonDirectionAttr:$comparison_direction,
    OptionalAttr<HLO_ComparisonTypeAttr>:$compare_type
  );
  let results = (outs HLO_PredTensor);

  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      "::mlir::mhlo::ComparisonDirection":$comparison_direction,
      CArg<"::mlir::mhlo::ComparisonType",
      "::mlir::mhlo::ComparisonType::NOTYPE">:$compare_type)>,
  ];

  let hasCustomHLOConverter = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

//===----------------------------------------------------------------------===//
// MHLO Slice definitions.
//===----------------------------------------------------------------------===//

def HLO_SliceOp: HLO_Op<
      "slice",
      [NoSideEffect, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def HLO_DynamicSliceOp: HLO_Op<"dynamic-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>,
       InferTensorType]> {
  let summary = "Dynamic Slice operator";
  let description = [{
    Extracts a sub-array from the input array at dynamic start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicslice.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    Variadic<HLO_ScalarIntTensor>:$start_indices,
    I64ElementsAttr:$slice_sizes
  );

  let results = (outs HLO_Tensor:$result);
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def HLO_DynamicUpdateSliceOp: HLO_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "update", "result"]>,
       AllShapesMatch<["operand", "result"]>]> {
  let summary = "Dynamic Update Slice operator";
  let description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$update,
    Variadic<HLO_ScalarIntTensor>:$start_indices
  );
  let results = (outs HLO_Tensor:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// MHLO Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormGradOp : HLO_Op<"batch_norm_grad", [NoSideEffect,
    AllShapesMatch<["scale", "mean", "variance", "grad_scale",
        "grad_offset"]>,
    AllShapesMatch<["operand", "grad_output"]>,
    AllElementTypesMatch<["operand", "grad_scale", "grad_offset"]>,
    AllTypesMatch<["operand", "grad_operand"]>]> {
  let summary = "Batch Normalization Gradient";
  let description = [{
    Calculates gradients of batch norm.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormgrad
  }];

  let arguments = (ins
    RankedTensorOf<[AnyFloat]>:$operand,
    1DTensorOf<[AnyFloat]>:$scale,
    1DTensorOf<[AnyFloat]>:$mean,
    1DTensorOf<[AnyFloat]>:$variance,
    RankedTensorOf<[AnyFloat]>:$grad_output,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs Variadic<HLO_TensorOrToken>);
  let results = (outs
      RankedTensorOf<[AnyFloat]>:$grad_operand,
      1DTensorOf<[AnyFloat]>:$grad_scale,
      1DTensorOf<[AnyFloat]>:$grad_offset);

  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_BatchNormInferenceOp : HLO_Op<"batch_norm_inference",
    [NoSideEffect, AllTypesMatch<["operand", "result"]>,
    AllShapesMatch<["scale", "offset", "mean", "variance"]>]> {
  let summary = "Batch Normalization for Inference";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];

  let arguments = (ins
    RankedTensorOf<[AnyFloat]>:$operand,
    1DTensorOf<[AnyFloat]>:$scale,
    1DTensorOf<[AnyFloat]>:$offset,
    1DTensorOf<[AnyFloat]>:$mean,
    1DTensorOf<[AnyFloat]>:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs RankedTensorOf<[AnyFloat]>:$result);

  let hasVerifier = 1;
}

def HLO_BatchNormTrainingOp : HLO_Op<"batch_norm_training",
    [NoSideEffect, AllTypesMatch<["operand", "output"]>,
    AllElementTypesMatch<["operand", "batch_mean", "batch_var"]>,
    AllShapesMatch<["scale", "offset", "batch_mean", "batch_var"]>]> {
  let summary = "Batch Normalization for Training";
  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnormtraining
  }];

  let arguments = (ins
    RankedTensorOf<[AnyFloat]>:$operand,
    1DTensorOf<[AnyFloat]>:$scale,
    1DTensorOf<[AnyFloat]>:$offset,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
      RankedTensorOf<[AnyFloat]>:$output,
      1DTensorOf<[AnyFloat]>:$batch_mean,
      1DTensorOf<[AnyFloat]>:$batch_var);

  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

def HLO_BitcastConvertOp : HLO_ShapedInterfaceOp<"bitcast_convert",
    [NoSideEffect]> {
  let summary = "BitcastConvert operator";
  let description = [{
    Similar to a 'tf.bitcast' in TensorFlow, performs an element-wise bitcast
    operation from a data shape to a target shape. The dimensions must match,
    and the conversion is an element-wise one. Bitcast is implemented as a
    low-level cast, so machines with different floating-point representations
    will give different results.

    See https://www.tensorflow.org/xla/operation_semantics#bitcastconverttype.
  }];

  let arguments = (ins HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
}

def HLO_BroadcastOp : HLO_ShapedInterfaceOp<"broadcast",
    [NoSideEffect, SameOperandsAndResultElementType, InferTensorType]> {
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
    HLO_Tensor:$operand,
    I64ElementsAttr:$broadcast_sizes
  );

  let results = (outs HLO_Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_BroadcastInDimOp : HLO_Op<"broadcast_in_dim",
      [NoSideEffect, SameOperandsAndResultElementType]> {
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
    HLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs HLO_StaticShapeTensor);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  // Only handles a static subset of the legacy format.
  let hasCustomHLOConverter = 1;
}

def HLO_DynamicBroadcastInDimOp : HLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim", [NoSideEffect]> {
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
    must be disjoint.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_DimensionTensor:$output_dimensions,
    BroadcastDimAttr:$broadcast_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_expanding_dimensions,
    OptionalAttr<BroadcastDimAttr>:$known_nonexpanding_dimensions
  );

  let results = (outs HLO_Tensor);

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

// Note: There is no HLO_CallOp because the standard call operation mlir::func::CallOp
// is used instead. A mlir::func::CallOp is exported to a HLO call instruction
// directly.

def HLO_CholeskyOp : HLO_Op<"cholesky",
      [NoSideEffect, SameOperandsAndResultElementType]> {
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
    HLO_FpOrComplexTensor:$a,
    DefaultValuedAttr<BoolAttr, "false">:$lower
  );

  let hasVerifier = 1;
  let results = (outs HLO_FpOrComplexTensor);
}

def HLO_ClampOp : HLO_ShapedInterfaceOp<"clamp", [NoSideEffect,
  SameOperandsAndResultElementType, HLO_BroadcastingElementwise]> {
  let summary = "Clamp operator";
  let description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.
  }];

  let arguments = (ins
    HLO_Tensor:$min,
    HLO_Tensor:$operand,
    HLO_Tensor:$max
  );
  let results = (outs HLO_Tensor);

  let hasVerifier = 1;
}

def HLO_ConcatenateOp : HLO_ShapedInterfaceOp<"concatenate",
    [NoSideEffect, SameOperandsAndResultElementType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "XLA's concatenate op";
  let description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$val,
    I64Attr: $dimension
  );

  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_CollectivePermuteOp: HLO_Op<"collective_permute",
    [NoSideEffect, SameOperandsAndResultType]> {
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
    HLO_Tensor:$operand,
    I64ElementsAttr:$source_target_pairs
  );
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;
}

def HLO_ConvOp : HLO_Op<"convolution", [NoSideEffect]> {
  let summary = "Convolution operator";
  let description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];
  let arguments = !con(
    (ins
       HLO_Tensor:$lhs,
       HLO_Tensor:$rhs),
    ConvolutionAttributes.attributes);

  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;

  code extraClassDeclaration = [{
    bool hasWindowReversal() {
      auto reversal = window_reversalAttr();
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

def HLO_CopyOp: HLO_Op<"copy", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Copy operator";
  let description = [{
    Returns a copy of `operand`.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
  let hasFolder = 1;
}

def HLO_CrossReplicaSumOp : HLO_Op<"cross-replica-sum",
    [NoSideEffect, SameOperandsAndResultType]> {
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
    HLO_Tensor:$operand,
    I64ElementsAttr:$replica_groups
  );

  let results = (outs HLO_Tensor);
}

def HLO_CustomCallOp: HLO_Op<"custom_call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "CustomCall operator";
  let description = [{
    A custom call invokes code external to XLA. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific. For example, in the CPU
    backend, a call instruction is emitted which targets a symbol with the name
    `call_target_name`.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.
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
  }];
  let arguments = (ins
    Variadic<HLO_TensorOrTokenOrTuple>:$operands,
    StrAttr:$call_target_name,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    // TODO(b/189822916): Remove this field when all clients are migrated to
    // the status-returning API.
    DefaultValuedAttr<
        HLO_CustomCallApiVersionAttr,
        "::mlir::mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL">:
        $api_version,
    DefaultValuedAttr<HLO_FlatSymbolRefArrayAttr, "{}">:$called_computations,
    OptionalAttr<HLO_ArrayOfLayoutAttr>:$operand_layouts,
    OptionalAttr<HLO_ArrayOfLayoutAttr>:$result_layouts
  );
  let results = (outs Variadic<HLO_TensorOrTokenOrTuple>);
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_DotOp: HLO_Op<"dot",
    [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Dot operator";
  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
  let arguments = (
    ins HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    HLO_PrecisionConfigAttr:$precision_config
  );
  let results = (outs HLO_Tensor);
  // Dot op required custom exporter to pass the preferred element type
  // to Xla builder.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_DotGeneralOp: HLO_Op<"dot_general", [NoSideEffect]> {
  let summary = "General Dot operator";
  let description = [{
    Performs general dot products between vectors, vector/matrix and
    matrix/matrix multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dotgeneral.
  }];
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    DotDimensionNumbers:$dot_dimension_numbers,
    HLO_PrecisionConfigAttr:$precision_config
  );

  let results = (outs HLO_Tensor);
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

def HLO_EinsumOp: HLO_Op<"einsum", [NoSideEffect]>, BASE_EinsumOp {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def HLO_UnaryEinsumOp: HLO_Op<"unary_einsum", [NoSideEffect]>, BASE_EinsumOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    StrAttr:$einsum_config
  );

  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;

  // UnaryEinsumOp is unconditionally canonicalized to the binary EinsumOp so
  // the HLO converter shouldn't be invoked.
  let hasCustomHLOConverter = 1;
}

def HLO_FftOp: HLO_Op<"fft", [NoSideEffect]> {
  let summary = "Fast fourier transform operator";
  let description = [{
    Returns the fast-fourier-transform of the input array.

    See
    https://www.tensorflow.org/xla/operation_semantics#fft.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_FftTypeAttr: $fft_type,
    I64ElementsAttr:$fft_length
  );

  let hasVerifier = 1;
  let results = (outs HLO_Tensor);
}

def HLO_GatherOp: HLO_Op<"gather", [InferTensorTypeWithReify, NoSideEffect]> {
  let summary = "Gather operator";
  let description = [{
    Stitches together several slices of `operand` from offsets specified in
    `start_indices` (each slice at a potentially different runtime offset).

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    GatherDimensionNumbers:$dimension_numbers,
    I64ElementsAttr:$slice_sizes,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted
  );

  let results = (outs HLO_Tensor);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_GetDimensionSizeOp: HLO_Op<"get_dimension_size", [NoSideEffect]> {
  let summary = "GetDimensionSize operator";
  let description = [{
    Returns the size of the given dimension of the operand.

    See
    https://www.tensorflow.org/xla/operation_semantics#getdimensionsize.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$dimension
  );
  // TODO(hinsu): Allow 64-bit result types once XLA HLO dialect based on the
  // XLA semantics is available. This limitation is because of the current XLA
  // implementation.
  let results = (outs I32Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_MapOp: HLO_Op<"map",
      [RecursiveSideEffects, SameOperandsElementType,
       SameOperandsAndResultShape, SingleBlockImplicitTerminator<"ReturnOp">]> {
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
    Variadic<HLO_Tensor>:$operands,
    I64ElementsAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$computation);
  let results = (outs HLO_Tensor);
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_ReshapeOp: HLO_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Reshape operator";
  let description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];
  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_StaticShapeTensor);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;

  let hasCustomHLOConverter = 1;
}

def HLO_DynamicReshapeOp: HLO_ShapedInterfaceOp<"dynamic_reshape", [NoSideEffect]> {
  let summary = "Reshape a tensor to a given, possibly dynamic, shape.";
  let description = [{
    Reshapes `operand` to `output_shape`.

    Requires:
    - The length of `output_shape` is equal to the rank of `result`.
    - The number of elements in `operand` (that is, the product of extents of
      its shape) is equal to the number of elements in `output_shape` (that is,
      the product of values in `output_shape`).
  }];

  let arguments = (ins HLO_Tensor:$operand, HLO_DimensionTensor:$output_shape);
  let results = (outs HLO_Tensor:$result);

  let hasCanonicalizer = 1;
  // Cannot be exported to legacy formats.
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_ScatterOp: HLO_Op<"scatter", [RecursiveSideEffects]> {
  let summary = "Scatter operator";
  let description = [{
    Generates a result which is the value of the input array `operand`,
    with several slices (at indices specified by `scatter_indices`)
    updated with the values in `updates` using `update_computation`.

    See https://www.tensorflow.org/xla/operation_semantics#scatter.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    TensorOf<[AnyInteger, Index]>:$scatter_indices,
    HLO_Tensor:$updates,
    ScatterDimensionNumbers:$scatter_dimension_numbers,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted,
    DefaultValuedAttr<BoolAttr, "false">:$unique_indices
  );

  let regions = (region SizedRegion<1>:$update_computation);

  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_SelectOp: HLO_Op<"select", [NoSideEffect, HLO_BroadcastingElementwise,
    InferTensorTypeWithReify]> {
  let summary = "Select operator";
  let description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`. All three operands must be of the same shape
    with the exception of `pred`, which may also be a scalar in which case it is
    broadcasted.

    See https://www.tensorflow.org/xla/operation_semantics#select.
  }];
  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_Tensor:$on_true,
    HLO_Tensor:$on_false
  );

  let results = (outs HLO_Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_SelectAndScatterOp: HLO_Op<"select_and_scatter",
      [RecursiveSideEffects]> {
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
    HLO_Tensor:$operand,
    HLO_Tensor:$source,
    HLO_Tensor:$init_value,
    OptionalAttr<I64ElementsAttr>:$window_dimensions,
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let regions = (region SizedRegion<1>:$select, SizedRegion<1>:$scatter);

  let results = (outs HLO_Tensor);

  let hasVerifier = 1;
  let hasCustomHLOConverter = 1;
}

def HLO_SetDimensionSizeOp: HLO_Op<"set_dimension_size", [NoSideEffect]> {
  let summary = "SetDimensionSize operator";
  let description = [{
    Sets the dynamic size of operand's given dimension. Pass through the operand
    as result, with dynamic dimension tracked by the compiler. Padded values
    will be ignored by downstream reduction ops.

    See https://www.tensorflow.org/xla/operation_semantics#setdimensionsize.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I32Tensor:$size,
    I64Attr:$dimension
  );
  let results = (outs HLO_Tensor);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_SortOp : HLO_Op<"sort", [RecursiveSideEffects,
                                 SameOperandsAndResultShape]> {
  let summary = "Sort operator";
  let description = [{
    Sorts the given `operands` at the given `dimension` with the given
    `comparator`.

    See https://www.tensorflow.org/xla/operation_semantics#sort.
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$operands,
    DefaultValuedAttr<I64Attr, "-1">:$dimension,
    DefaultValuedAttr<BoolAttr, "false">:$is_stable
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  // TODO(b/129422361): SortOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def HLO_ReverseOp: HLO_Op<"reverse",
      [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Reverse operator";
  let description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$dimensions
  );

  let results = (outs HLO_Tensor);

  let hasFolder = 1;
}

def HLO_PadOp: HLO_Op<"pad",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Pad operator";
  let description = [{
    Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration.

    See https://www.tensorflow.org/xla/operation_semantics#pad.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$padding_value,
    I64ElementsAttr: $edge_padding_low,
    I64ElementsAttr: $edge_padding_high,
    I64ElementsAttr: $interior_padding
  );

  let results = (outs HLO_Tensor);

  let description = [{
    Pads the `operand` according to TBD.
  }];

  // TODO(b/129422361): PadOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;

  let hasFolder = 1;
  let hasVerifier = 1;
}

def HLO_TraceOp: HLO_Op<"trace", []> {
  let summary = "Trace operator";
  let description = [{
    Emits a logging message `tag` with the `operand`.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    StrAttr:$tag
  );
  let hasCustomHLOConverter = 1;
}

def HLO_TransposeOp: HLO_ShapedInterfaceOp<"transpose",
      [NoSideEffect, SameOperandsAndResultElementType,
       InferTensorTypeWithReify]> {
  let summary = "Transpose operator";
  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I64ElementsAttr:$permutation
  );
  let results = (outs HLO_Tensor);

  let extraClassDeclaration = [{
    // Method from  InferTypeOpInterface interface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def HLO_TriangularSolveOp: HLO_Op<"triangular_solve",
    [NoSideEffect, SameOperandsAndResultElementType]> {
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
    HLO_FpOrComplexTensor:$a,
    HLO_FpOrComplexTensor:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    HLO_TransposeAttr:$transpose_a
  );
  let results = (outs HLO_FpOrComplexTensor);

  let hasVerifier = 1;
}

def HLO_ReduceWindowOp: HLO_Op<"reduce_window", [
      RecursiveSideEffects,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">
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
    Variadic<HLO_Tensor>:$operands,
    Variadic<HLO_Tensor>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<HLO_Tensor>);

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
    }]>
  ];

  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
  // TODO(hinsu): Implement custom printer and parser.

  let extraClassDeclaration = [{
     // Get the operation used for reduction applied to `result_index`th result.
     Operation *getReductionOp(int result_index);
  }];
}

def HLO_ReturnOp : HLO_Op<"return", [NoSideEffect, Terminator]> {
  let summary = [{
    The `hlo.return` operation terminates a region and returns values.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTokenOrTuple >:$results
  );

  // Disable conversion operator for return op as the op is not an actual XLA
  // instruction and is only used as a terminator for regions.
  let hasCustomHLOConverter = 1;

  // TODO(hinsu): Implement custom printer and parser.
}

def HLO_TorchIndexSelectOp : HLO_Op<"torch_index_select", [NoSideEffect]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$index,
    I64Attr:$dim,
    I64Attr:$batch_dims
  );

  let results = (outs HLO_Tensor);

  // TODO(hinsu): Canonicalize to lower this client side HLO op to server
  // side HLO ops.
}

def HLO_OptimizationBarrierOp : HLO_Op<"optimization_barrier",
      [NoSideEffect, HLO_PairwiseSameOperandAndResultType]> {
  let summary = [{
    The `hlo.optimization_barrier` op blocks optimizations.
  }];
  let description = [{
    Blocks any optimization pass from moving computations across the barrier.

    Ensures that all inputs are evaluated before any operators that depend on the barrier's outputs.
    See
    https://www.tensorflow.org/xla/operation_semantics#optimizationbarrier
  }];

  let arguments = (ins Variadic<HLO_TensorOrToken>:$operand);

  let results = (outs Variadic<HLO_TensorOrToken>);

  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO RNG Operators.
//===----------------------------------------------------------------------===//

def HLO_RngUniformOp : HLO_Op<"rng_uniform", [InferTensorTypeWithReify, AllElementTypesMatch<["a", "b", "result"]>]> {
  let summary = "RNG with uniform distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the uniform distribution over the interval `[a,b)`. The parameters
    and output element type have to be a boolean type, an integral type or a
    floating point types, and the types have to be consistent.

    See https://www.tensorflow.org/xla/operation_semantics#rnguniform.
  }];
  let arguments = (ins
    0DTensorOf<[HLO_Pred, HLO_Int, AnyFloat]>:$a,
    0DTensorOf<[HLO_Pred, HLO_Int, AnyFloat]>:$b,
    HLO_DimensionTensor:$shape
  );

  let results = (outs HLO_PredIntOrFpTensor:$result);

  let hasCustomHLOConverter = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(::mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_RngNormalOp : HLO_Op<"rng_normal", [InferTensorTypeWithReify, AllElementTypesMatch<["mu", "sigma", "result"]>]> {
  let summary = "RNG with normal distribution.";
  let description = [{
    Constructs an output of a given shape with random numbers generated
    following the normal distribution with parameters `mu` and `sigma`. The
    parameters and output shape have to have a floating point elemental type.
    The parameters furthermore have to be scalar valued.

    See https://www.tensorflow.org/xla/operation_semantics#rngnormal.
  }];
  let arguments = (ins
    0DTensorOf<[AnyFloat]>:$mu,
    0DTensorOf<[AnyFloat]>:$sigma,
    HLO_DimensionTensor:$shape
  );

  let results = (outs HLO_FpTensor:$result);

  let hasCustomHLOConverter = 1;

  let extraClassDeclaration = [{
    // Returns whether the return types are compatible.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(::mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_RngBitGeneratorOp : HLO_Op<"rng_bit_generator", [NoSideEffect]> {
  let summary = "Uniform random number generator operator";
  let description = [{
    Returns an output with a given shape filled with uniform random bits using
    the specified algorithm (or backend default) and returns an updated state
    (with the same shape as initial state) and the generated random data.

    See
    https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator.
  }];
  let arguments = (ins
    HLO_RngAlgorithmAttr:$rng_algorithm,
    HLO_IntOrFpTensor:$initial_state
  );

  let results = (outs
      HLO_IntOrFpTensor:$output_state,
      HLO_IntOrFpTensor:$output
      );

  let hasVerifier = 1;
  // TODO(jpienaar): This should not be needed.
  let hasCustomHLOConverter = 1;
}

def HLO_XlaRngGetAndUpdateStateOp: HLO_Op<"xla.rng_get_and_update_state", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
def HLO_UniformQuantizeOp : HLO_UnaryElementwiseOp<"uniform_quantize",
      [NoSideEffect], TensorOf<[F32, BF16, HLO_QuantizedInt]>,
      HLO_QuantizedIntTensor> {
  let summary = "Uniform quantize operator";
  let description = [{
    Converts floating point tensors or uniform quantized integer tensors to
    uniform quantized integer tensors according to the quantization parameters
    defined by the output type.
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def HLO_UniformDequantizeOp : HLO_UnaryElementwiseOp<"uniform_dequantize",
      [NoSideEffect], HLO_QuantizedIntTensor, TensorOf<[F32, BF16]>> {
  let summary = "Uniform dequantize operator";
  let description = [{
    Converts quantized array of integers to floating-points according to the
    quantization parameters defined by the input type.
  }];

  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  let hasCustomHLOConverter = 1;
}

def HLO_FusionOp : HLO_Op<"fusion", []> {
  let summary = "Fusion operator";
  let description = [{
    Models the fusion instruction.

    A fusion op is consists of a group of basic ops (represented as a region
    attached to it). It serves as a hint to the backend that it is beneficial
    to emit the contained ops into a single loop nest or kernel.
  }];
  let regions = (region SizedRegion<1>:$fused_computation);

  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$operands,
    OptionalAttr<HLO_FusionKindAttr>:$fusion_kind
  );

  let results = (outs
    Variadic<HLO_TensorOrTuple>:$results
  );

  // FusionOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

// This is an op for purposes internal to XLA/GPU.
def HLO_BitcastOp : HLO_Op<"bitcast", [NoSideEffect]> {
  let summary = "Bitcast operator";
  let description = [{
    This op changes the shape of the input in the way that the physical
    arrangement of elements are unchanged.

    However, the op needs layout information to make sense of "physical
    arrangement of elements". Layout support in MHLO is currently under
    exploration.
  }];
  let arguments = (ins HLO_Tensor:$operand);
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
}

def HLO_ReducePrecisionOp :
    HLO_Op<"reduce_precision", [SameOperandsAndResultShape, SameOperandsAndResultElementType]> {
  let summary = "Reduce precision operator";
  let description = [{
    Models the effect of converting floating - point values to a lower -
    precision format(such as IEEE - FP16) and back to the original
    format. The number of exponent and mantissa bits in the lower -
    precision format can be specified arbitrarily,
    although all bit sizes may not be supported on all hardware
    implementations.

    See https://www.tensorflow.org/xla/operation_semantics#reduceprecision.
  }];
  let arguments = (ins
    HLO_FpTensor:$operand,
    I32Attr:$exponent_bits,
    I32Attr:$mantissa_bits
  );
  let hasVerifier = 1;
  let results = (outs HLO_FpTensor:$output);
}

def HLO_RealDynamicSliceOp: HLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "Real Dynamic Slice operator";
  let description = [{
    The dynamic shape version of SliceOp. Extracts a sub-array from the input
    array according to start_indices, limit_indices and strides. Expect
    start_indices/limit_indices/strides to be statically shaped and matching
    the rank of the input.
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_DimensionTensor:$start_indices,
    HLO_DimensionTensor:$limit_indices,
    HLO_DimensionTensor:$strides
  );
  let results = (outs HLO_Tensor:$result);
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_DynamicPadOp: HLO_ShapedInterfaceOp<"dynamic_pad",
      [NoSideEffect, AllElementTypesMatch<["operand", "padding_value", "result"]>,
      AllTypesMatch<["edge_padding_low", "edge_padding_high", "interior_padding"]>]> {
  let summary = "Dynamic Pad operator";
  let description = [{
    The dynamic shape version of PadOp. Pads the edges of `operand` with the
    `padding_value` and according to the passed configuration. Expect
    edge_padding_low/edge_padding_high/interior_padding to be statically shaped
    and matching the rank of the input.
    See
    https://www.tensorflow.org/xla/operation_semantics#pad
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$padding_value,
    HLO_DimensionTensor:$edge_padding_low,
    HLO_DimensionTensor:$edge_padding_high,
    HLO_DimensionTensor:$interior_padding
  );
  let results = (outs HLO_Tensor:$result);
  let description = [{
    Dynamically Pads the `operand`, with amount of padding added at
    low-end/high-end/interior is passed through input tensors.
  }];
  let hasCanonicalizer = 1;
  let hasCustomHLOConverter = 1;
  let hasVerifier = 1;
}

def HLO_DynamicGatherOp: HLO_Op<"dynamic_gather",
                                [InferTensorTypeWithReify, NoSideEffect]> {
  string summary = "Dynamic Gather operator";
  string description = [{
    The dynamic shape version of GatherOp. Stitches together several slices of
    an input array.
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_IntTensor:$start_indices,
    HLO_IntTensor:$slice_sizes,
    GatherDimensionNumbers:$dimension_numbers,
    DefaultValuedAttr<BoolAttr, "false">:$indices_are_sorted
  );
  let results = (outs HLO_Tensor);

  let hasCustomHLOConverter = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return succeeded(mlir::verifyCompatibleShapes(l, r));
    }
  }];
}

def HLO_DynamicConvOp : HLO_Op<"dynamic_conv", [NoSideEffect]> {
  let summary = "Dynamic Convolution operator";
  let description = [{
    The dynamic shape version of ConvOp. Computes a convolution with dynamic padding.
  }];

  let arguments = !con(
    (ins
       HLO_Tensor:$lhs,
       HLO_Tensor:$rhs,
       HLO_Tensor:$d_padding),
    ConvolutionAttributes.attributes);
  let results = (outs HLO_Tensor);
  let hasCustomHLOConverter = 1;
}

def HLO_PrintOp : HLO_Op<"print", []> {
  let summary = "Print operation";
  let description = [{
    PrintOp is used to print tensor, which can be used to debug.
  }];

  let arguments = (ins
      HLO_Tensor:$operand
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let hasCustomHLOConverter = 1;
}

def HLO_ComputeReshapeShapeOp :
    HLO_Op<"compute_reshape_shape", [NoSideEffect]> {
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
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs 1DTensorOf<[AnyInteger, Index]>:$result);

  let assemblyFormat = "$num_elements `,` $dynamic_shape attr-dict `:` type($num_elements) `,` type($dynamic_shape) `->` type($result)";
  let hasCustomHLOConverter = 1;
}

def HLO_CstrReshapableOp :
    HLO_Op<"cstr_reshapable", [NoSideEffect]> {
  string summary = "Compute input for reshape with any dynamic dim resolved";

  string description = [{
    This operation creates a witness on the constraint that a given shape would
    be a valid reshape for the given number of elements.

    ```
    %0 = mhlo.cstr_reshapable 12, [2, -1] -> success
    %1 = mhlo.cstr_reshapable 13, [2, -1] -> failure
    ```
  }];

  let arguments = (ins Index:$num_elements, 1DTensorOf<[AnyInteger, Index]>:$dynamic_shape);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$num_elements `,` $dynamic_shape attr-dict `:` type($num_elements) `,` type($dynamic_shape)";

  let hasCustomHLOConverter = 1;
}

#endif // HLO_OPS
