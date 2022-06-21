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

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def HLO_Dialect : Dialect {
  let name = "mhlo";
  let cppNamespace = "::mlir::mhlo";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.td"

def HLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def HLO_SInt : SignlessIntOfWidths<[4, 8, 16, 32, 64]>;
def HLO_UInt : UnsignedIntOfWidths<[4, 8, 16, 32, 64]>;
def HLO_Int : AnyTypeOf<[HLO_SInt, HLO_UInt]>;

def HLO_Float : AnyTypeOf<[F16, F32, F64, BF16]>;

def HLO_Complex : Complex<AnyTypeOf<[F32, F64]>>;

//===----------------------------------------------------------------------===//
// Quantized element type definitions.
//===----------------------------------------------------------------------===//

// TODO(b/230381284): Upstream width-specific uniform quantized element types.
class UniformQuantizedSignedInt<int width>
  : Type<And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # width>,
              CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                    ".isSigned()">]>,
    "QI" # width # " type"> {
  string name = "UniformQuantizedSignedInt";
  int bitwidth = width;
}

class UniformQuantizedUnsignedInt<int width>
  : Type<And<[CPred<"$_self.isa<mlir::quant::UniformQuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::UniformQuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # width>,
              CPred<"!$_self.cast<mlir::quant::UniformQuantizedType>()" #
                    ".isSigned()">]>,
    "QUI" # width # " type"> {
  string name = "UniformQuantizedUnsignedInt";
  int bitwidth = width;
}

class UniformQuantizedSignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, UniformQuantizedSignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized signed " #
              "integer">;

class UniformQuantizedUnsignedIntOfWidths<list<int> widths> :
    AnyTypeOf<!foreach(w, widths, UniformQuantizedUnsignedInt<w>),
              !interleave(widths, "/") # "-bit uniform quantized unsigned " #
              "integer">;

// Integer-based uniform quantized types. The definitions can be used to specify
// operand's tensor types.
def HLO_QuantizedSignedInt : UniformQuantizedSignedIntOfWidths<[4, 8, 16, 32]>;
def HLO_QuantizedUnsignedInt : UniformQuantizedUnsignedIntOfWidths<[4, 8, 16, 32]>;
def HLO_QuantizedInt :
    AnyTypeOf<[HLO_QuantizedSignedInt, HLO_QuantizedUnsignedInt]>;

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
defvar BroadcastDimAttr = I64ElementsAttr;

//===----------------------------------------------------------------------===//
// MHLO on tensors type definitions.
//===----------------------------------------------------------------------===//

// Token type.
def HLO_Token : Type<CPred<"$_self.isa<TokenType>()">, "token">;

// Any integer tensor types
def HLO_IntTensor : TensorOf<[HLO_Int]>;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
def HLO_ScalarIntTensor : 0DTensorOf<[HLO_Int]>;

// Any floating-point tensor types
def HLO_FpTensor : TensorOf<[HLO_Float]>;

// Any quantized integer tensor types
def HLO_QuantizedIntTensor : TensorOf<[HLO_QuantizedInt]>;

def HLO_PredTensor : TensorOf<[HLO_Pred]>;

def HLO_Tensor : TensorOf<[HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt]>;

def HLO_ComplexTensor : TensorOf<[HLO_Complex]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Tuple]>;

def HLO_TensorOrToken : AnyTypeOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrTokenOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Token, HLO_Tuple]>;

def HLO_DimensionValue : AnyTypeOf<[Index, HLO_Int]>;

// Dynamic representation of a shape vector as a tensor.
def HLO_DimensionTensor : 1DTensorOf<[HLO_DimensionValue]>;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def HLO_StaticShapeTensor : StaticShapeTensorOf<[
      HLO_Float, HLO_Pred, HLO_Int, HLO_Complex, HLO_QuantizedInt]>;

//===----------------------------------------------------------------------===//
// MHLO on tensors combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : TensorOf<[HLO_Int, HLO_Float]>;

// Any integer or predicate tensor types
def HLO_PredOrIntTensor : TensorOf<[HLO_Pred, HLO_Int]>;

// Any floating-point or complex tensor types
def HLO_FpOrComplexTensor : TensorOf<[HLO_Float, HLO_Complex]>;

// Any int, floating-point or complex tensor types
def HLO_IntFpOrComplexTensor : TensorOf<[HLO_Int, HLO_Float, HLO_Complex]>;

// Any pred, int or floating-point tensor types
def HLO_PredIntOrFpTensor : TensorOf<[HLO_Pred, HLO_Int, HLO_Float]>;

// A layout attribute (1D tensor of index type)
def HLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{$_self.cast<::mlir::DenseIntElementsAttr>().getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def HLO_ArrayOfLayoutAttr : TypedArrayAttrBase<HLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def HLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}


//===----------------------------------------------------------------------===//
// Common convolution attributes
//===----------------------------------------------------------------------===//

// TODO(b/129153247) See if it's possible to also validate the size.
def HLO_PrecisionConfigAttr:
    OptionalAttr<
          TypedArrayAttrBase<HLO_PrecisionAttr, "Precision Config attribute">>;

def BoolElementsAttr :
    ElementsAttrBase<
      And<[CPred<"$_self.isa<::mlir::DenseIntOrFPElementsAttr>()">,
           CPred<"$_self.cast<::mlir::DenseIntOrFPElementsAttr>().getType().getElementType().isInteger(1)">]>,
      "constant boolean vector/tensor attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];

  let convertFromStorage = "$_self";
}

def ConvolutionAttributes {
  dag attributes = (ins
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: two zeros for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    // Default value: false for each of the spatial dimension.
    OptionalAttr<BoolElementsAttr>:$window_reversal,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    HLO_PrecisionConfigAttr:$precision_config
  );
}

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

class HLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::mhlo::OpTrait";
}

// An operation that is essentially element-wise but may implement broadcasting
// semantics.
def HLO_BroadcastingElementwise : HLO_NativeOpTrait<"BroadcastingElementwise">;

// Op has pairwise operand and result type matching: the number of operands
// must be equal to the number of results and the type of ith operand must
// match the type of ith result.
// TODO(b/195086460) Promote this to be an mlir trait and remove it here.
def HLO_PairwiseSameOperandAndResultType :
  HLO_NativeOpTrait<"PairwiseSameOperandAndResultType">;

// Op has operand and result types compatible with each other according to
// the rules implemented in isCompatibleForMhloTypeInference, which account for
// special properties dynamism, quantization and sparsity.
def HLO_CompatibleOperandsAndResultType : TraitList<
  // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
  // support quantization or sparsity.
  [
    InferTypeOpInterface,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
    HLO_NativeOpTrait<"CompatibleOperandsAndResultType">
  ]>;

#endif // HLO_OPS_BASE
