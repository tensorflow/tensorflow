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

// This is the utils file for the HLO dialect.

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_UTILS
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_UTILS

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

def NullArrayAttr : NativeCodeCall<"ArrayAttr()">;

def NullChannelHandleAttr : NativeCodeCall<"mhlo::ChannelHandleAttr()">;
def StableHLO_NullChannelHandleAttr : NativeCodeCall<"stablehlo::ChannelHandleAttr()">;

def CastIntElementsAttr : NativeCodeCall<"cast<DenseIntElementsAttr>($0)">;

class ConstantSplat<string value> : NativeCodeCall<
    "hlo::getSplat(&$_builder, $0, " # value # ")">;

class StableHLO_ConstantLike<string value> : NativeCodeCall<
    "::mlir::stablehlo::getConstantLike($_builder, $_loc, " # value # ", $0)">;

def MHLO_ConstantLikeMaxFiniteValue : NativeCodeCall<
    "chlo::getConstantLikeMaxFiniteValue($_builder, $_loc, $0)">;

def MHLO_ConstantLikePosInfValue : NativeCodeCall<
    "chlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/false)">;

def MHLO_ConstantLikeNegInfValue : NativeCodeCall<
    "chlo::getConstantLikeInfValue($_builder, $_loc, $0, /*negative=*/true)">;

def NullDenseIntElementsAttr : NativeCodeCall<"DenseIntElementsAttr()">;

def NullDenseI64ArrayAttr : NativeCodeCall<"DenseI64ArrayAttr()">;

def BinBroadcastDimensions : NativeCodeCall<
    "hlo::getBroadcastDimensionsAttr(&$_builder, $0, $1)">;

def BinBroadcastDimensionsNonEmpty : NativeCodeCall<
    "hlo::getBroadcastDimensionsAttr(&$_builder, $0, $1, /*allow_empty=*/false)">;

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the value.
class GetScalarOfType<int value> : NativeCodeCall<
  "hlo::getScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

// Constraint that Attr has values [0, 1, ...].
def IdentityBroadcastDims : AttrConstraint<
    CPred<"hlo::isSequenceStartingWith0($_self)">>;

def ComplexElementType : Type<
  CPred<"isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Complex element type">;

def NonComplexElementType : Type<
  CPred<"!isa<ComplexType>(cast<ShapedType>($_self).getElementType())">,
  "Non-complex element type">;

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_UTILS
