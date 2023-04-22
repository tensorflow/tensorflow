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

def HLO_Dialect : Dialect {
  let name = "mhlo";
  let cppNamespace = "::mlir::mhlo";
}

include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.td"

def HLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def HLO_SInt : SignlessIntOfWidths<[8, 16, 32, 64]>;
def HLO_UInt : UnsignedIntOfWidths<[8, 16, 32, 64]>;
def HLO_Int : AnyTypeOf<[HLO_SInt, HLO_UInt]>;

def HLO_Complex : Complex<AnyTypeOf<[F32, F64]>>;

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
def HLO_FpTensor : TensorOf<[AnyFloat]>;

def HLO_PredTensor : TensorOf<[HLO_Pred]>;

def HLO_Tensor : TensorOf<[AnyFloat, HLO_Pred, HLO_Int, HLO_Complex]>;

def HLO_ComplexTensor : TensorOf<[HLO_Complex]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Tuple]>;

def HLO_TensorOrTokenOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Token, HLO_Tuple]>;

def HLO_DimensionValue : AnyTypeOf<[Index, HLO_Pred, HLO_Int]>;

// Dynamic representation of a shape vector as a tensor.
def HLO_DimensionTensor : 1DTensorOf<[HLO_DimensionValue]>;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def HLO_StaticShapeTensor : StaticShapeTensorOf<[
      AnyFloat, HLO_Pred, HLO_Int, HLO_Complex]>;

//===----------------------------------------------------------------------===//
// MHLO on tensors combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : TensorOf<[HLO_Int, AnyFloat]>;

// Any integer or predicate tensor types
def HLO_PredOrIntTensor : TensorOf<[HLO_Pred, HLO_Int]>;

// Any floating-point or complex tensor types
def HLO_FpOrComplexTensor : TensorOf<[AnyFloat, HLO_Complex]>;

// Any int, floating-point or complex tensor types
def HLO_IntFpOrComplexTensor : TensorOf<[HLO_Int, AnyFloat, HLO_Complex]>;

// Any pred, int or floating-point tensor types
def HLO_PredIntOrFpTensor : TensorOf<[HLO_Pred, HLO_Int, AnyFloat]>;

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
    // Default value: zero for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$padding,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$lhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$rhs_dilation,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<BoolElementsAttr>:$window_reversal,
    ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count,
    HLO_PrecisionConfigAttr:$precision_config
  );
}

class BASE_HLO_ConvOp {
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

#endif // HLO_OPS_BASE
