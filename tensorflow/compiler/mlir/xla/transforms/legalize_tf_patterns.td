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

// This is the legalization pattern definition file for TF to XLA.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/xla/ir/xla_ops.td"

def NullElementsAttr : NativeCodeCall<"ElementsAttr()">;

def HasNoUse: Constraint<
    CPred<"$0->use_begin() == $0->use_end()">, "has no use">;

//===----------------------------------------------------------------------===//
// BatchNorm op patterns.
//===----------------------------------------------------------------------===//

def FeatureDimension : NativeCodeCall<
    "getFeatureDimensionAttr($_builder, $0, $1)">;
def FalseBoolAttr : AttrConstraint<CPred<"!$_self.getValue()">>;

def : Pattern<
    (TF_FusedBatchNormOp:$root $x, $scale, $offset, $mean, $variance, $epsilon,
                               $data_format, FalseBoolAttr:$is_training),
    [(XLA_BatchNormInferenceOp $x, $scale, $offset, $mean, $variance,
                               $epsilon, (FeatureDimension $data_format, $x)),
     // We already guaranteed that the last four results has no use so it
     // does not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x)],
    [(HasNoUse $root__1), (HasNoUse $root__2),
     (HasNoUse $root__3), (HasNoUse $root__4)]>;

//===----------------------------------------------------------------------===//
// Bias op patterns.
//===----------------------------------------------------------------------===//
def Is1DShapeTensor
    : Type<CPred<"$_self.isa<RankedTensorType>() && "
                 "$_self.cast<RankedTensorType>().getRank() == 1">>;
def IsAtleast3DShapeTensor
    : Type<CPred<"$_self.isa<RankedTensorType>() && "
                 "$_self.cast<RankedTensorType>().getRank() > 2">>;

def BiasAddFeatureDimension : NativeCodeCall<
    "getBiasFeatureDimension($_builder, $0, $1)">;
def ValidBiasAddFeatureDimension : Constraint<
    CPred<"hasValidBiasFeatureDimension($0, $1, $2)">,
    "valid biasAdd feature dimension">;

def : Pat<(TF_BiasAddOp IsAtleast3DShapeTensor:$input, Is1DShapeTensor:$bias,
                        TF_ConvnetDataFormatAttr:$data_format),
          (XLA_AddOp $input, $bias,
              (BiasAddFeatureDimension $data_format, $input)),
          [(ValidBiasAddFeatureDimension $data_format, $input, $bias)]>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
//===----------------------------------------------------------------------===//

// Get the broadcast dimensions attribute from the binary operands.
def BinBroadcastDimensions : NativeCodeCall<
    "getBroadcastDimensionsAttr($_builder, $0, $1)">;

class DirectBinaryPat<Op FromOp, Op ToOp>
  : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
        (ToOp $l, $r, (BinBroadcastDimensions $l, $r))>;

foreach fromToBinPair = [[TF_AddOp, XLA_AddOp],
                         [TF_DivOp, XLA_DivOp],
                         [TF_MulOp, XLA_MulOp],
                         [TF_RealDivOp, XLA_DivOp],
                         [TF_SubOp, XLA_SubOp]] in
  def : DirectBinaryPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Identity op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_IdentityOp $op), (replaceWithValue $op)>;

//===----------------------------------------------------------------------===//
// Nullary op patterns.
//===----------------------------------------------------------------------===//

// TODO(riverriddle) Formalize a policy on converting opaque attributes.
def : Pat<(TF_ConstOp:$res ElementsAttr:$value), (XLA_ConstOp $value),
          [(AnyStaticShapeTensor $res)]>;

//===----------------------------------------------------------------------===//
// Relu op patterns.
//===----------------------------------------------------------------------===//

class ConstantSplat<string value> : NativeCodeCall<
    "getSplat($_builder, $0, " # value # ")">;

def : Pat<(TF_ReluOp AnyTensor:$input),
          (XLA_MaxOp (ConstantOp (ConstantSplat<"0"> $input)), $input,
                     (NullElementsAttr))>;

def : Pat<(TF_Relu6Op AnyTensor:$input),
          (XLA_ClampOp (ConstantOp (ConstantSplat<"0"> $input)), $input,
                       (ConstantOp (ConstantSplat<"6"> $input)))>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_ReshapeOp:$res AnyStaticShapeTensor:$arg, $ignored),
          (XLA_ReshapeOp $arg), [(AnyStaticShapeTensor $res)]>;

def : Pat<(TF_SqueezeOp AnyStaticShapeTensor:$arg, $ignored_dims),
          (XLA_ReshapeOp $arg)>;
