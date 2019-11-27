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
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class GetScalarOfType<int value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

//===----------------------------------------------------------------------===//
// BiasAddGrad op patterns.
//===----------------------------------------------------------------------===//

def GetBiasAddGradReductionIndices : NativeCodeCall<
  "GetBiasAddGradReductionIndices("
  "$0->getType().cast<RankedTensorType>().getRank(), $1, &$_builder)">;

def LowerBiasAddGradOp :
  Pat<(TF_BiasAddGradOp AnyRankedTensor:$out_backprop, $data_format),
      (TF_SumOp $out_backprop,
                (TF_ConstOp (GetBiasAddGradReductionIndices $out_backprop,
                                                            $data_format)),
                /*keep_dims=*/ConstBoolAttrFalse)>;

// Lowers SoftmaxCrossEntropyWithLogitsOp using simpler TensorFlow ops. The op
// computes loss and backprop of the loss with respect to 'features'.
//
// Softmax cross entropy loss is defined as follows:
//
//  loss = Sum(-labels * Log(Exp(features) / Sum(Exp(features)))
//  loss = Sum(-labels * LogSoftmax(features))
//
// Computing gradient of the loss with respect to features gives us,
//
//  backprop = (Exp(features) / Sum(Exp(features))) - labels
//  backprop = Softmax(features) - labels
//
// Computation of the reduction axis for the Sum op depends on whether the
// input is a scalar or not. Restrict pattern to ranked inputs so that input to
// the Sum op is also ranked.

// TODO(hinsu): Support scalar inputs by introducing reshape to 1D.
def NonScalarType : Type<Neg<HasAnyRankOfPred<[0]>>, "Non scalar type">;

def GetLastDimReductionAxis :
  NativeCodeCall<"GetI64ElementsAttr({-1}, &$_builder)">;

def LowerSoftmaxCrossEntropyWithLogitsOp : Pattern<
  (TF_SoftmaxCrossEntropyWithLogitsOp AnyRankedTensor:$features,
                                      AnyRankedTensor:$labels),
  [(TF_SumOp (TF_MulOp:$sum_input (TF_NegOp $labels),
                                  (TF_LogSoftmaxOp $features)),
             (TF_ConstOp (GetLastDimReductionAxis)),
             /*keep_dims=*/ConstBoolAttrFalse),
   (TF_SubOp (TF_SoftmaxOp $features), $labels)],
  [(NonScalarType $features), (NonScalarType $labels)]>;

//===----------------------------------------------------------------------===//
// Difference op patterns.
//===----------------------------------------------------------------------===//

def ComplexTensor   : TensorOf<[AnyComplex]>;
def RealTensor   : TensorOf<[AnyInteger, AnyFloat]>;

def : Pat<(TF_SquareOp $val), (TF_MulOp $val, $val)>;

def : Pat<(TF_SquaredDifferenceOp RealTensor: $lhs, RealTensor:$rhs),
          (TF_SquareOp (TF_SubOp $lhs, $rhs))>;

def : Pat<(TF_SquaredDifferenceOp ComplexTensor: $lhs, ComplexTensor:$rhs),
          (TF_MulOp (TF_SubOp:$diff $lhs, $rhs), (TF_ConjOp $diff))>;

//===----------------------------------------------------------------------===//
// DivNoNan and MulNonNan op patterns.
//===----------------------------------------------------------------------===//

class BinaryNoNanPat<Op FromOp, Op ToOp>
  : Pat<(FromOp $l, $r),
        (TF_SelectV2Op (TF_EqualOp $r,
                                   (TF_ConstOp:$zero (GetScalarOfType<0> $r)),
                       /*incompatible_shape_error*/ConstBoolAttrTrue),
           $zero, (ToOp $l, $r))>;

foreach fromToBinPair = [[TF_DivNoNanOp, TF_DivOp],
                         [TF_MulNoNanOp, TF_MulOp]] in
  def : BinaryNoNanPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Fill op patterns.
//===----------------------------------------------------------------------===//

def LowerFillOp : Pat<(TF_FillOp $dims, $value),
                      (TF_BroadcastToOp $value, $dims)>;

//===----------------------------------------------------------------------===//
// L2Loss op patterns.
//===----------------------------------------------------------------------===//

def GetAllAxes : NativeCodeCall<
  "GetI64ElementsAttrForSeq("
  "0, $0->getType().cast<RankedTensorType>().getRank(), &$_builder)">;

// L2Loss is lowered using the formula,
// L2Loss(input) = Sum(input * input) / 2

// TODO(hinsu): Support unranked tensors once b/144593778 is resolved to not
// cause result type mismatch.
def LowerL2LossOp :
  Pat<(TF_L2LossOp AnyRankedTensor:$input),
      (TF_DivOp
        (TF_SumOp (TF_MulOp $input, $input),
                  (TF_ConstOp (GetAllAxes $input)),
                  /*keep_dims=*/ConstBoolAttrFalse),
        (TF_ConstOp (GetScalarOfType<2> $input)))>;

//===----------------------------------------------------------------------===//
// Pad op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_PadOp TensorOf<[AnyInteger, AnyFloat]>:$input, $paddings),
          (TF_PadV2Op $input, $paddings,
             (TF_ConstOp (GetScalarOfType<0> $input)))>;

//===----------------------------------------------------------------------===//
// Rsqrt op patterns.
//===----------------------------------------------------------------------===//

// RsqrtGrad(lhs, rhs) = (lhs * lhs * lhs) * (rhs / -2)
def : Pat<(TF_RsqrtGradOp $lhs, $rhs),
          (TF_MulOp (TF_MulOp (TF_MulOp $lhs, $lhs), $lhs),
                    (TF_DivOp $rhs,
                              (TF_ConstOp (GetScalarOfType<-2> $rhs))))>;


//===----------------------------------------------------------------------===//
// TanhGrad op patterns.
//===----------------------------------------------------------------------===//

// grad = dy * (1 - y**2)

// TODO(hinsu): Support complex input types.
def LowerTanhGradOp :
  Pat<(TF_TanhGradOp TF_FpTensor:$y, TF_FpTensor:$dy),
      (TF_MulOp $dy,
                (TF_SubOp (TF_ConstOp (GetScalarOfType<1> $y)),
                          (TF_SquareOp $y)))>;

//===----------------------------------------------------------------------===//
// ZerosLike op patterns.
//===----------------------------------------------------------------------===//

def CreateTFShapeOp : NativeCodeCall<
    "$_builder.create<TF::ShapeOp>($0->getLoc(), $1, $2)">;

// TODO(hinsu): Support inputs of TensorList types.
def LowerZerosLikeOp :
  Pat<(TF_ZerosLikeOp:$src_op TensorOf<[AnyInteger, AnyFloat]>:$input),
      (TF_BroadcastToOp (TF_ConstOp (GetScalarOfType<0> $input)),
                        (CreateTFShapeOp $src_op, $input, /*use 32bit*/ConstBoolAttrFalse))>;
