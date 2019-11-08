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
// Pad op patterns.
//===----------------------------------------------------------------------===//

def ZeroDenseElementsAttr : NativeCodeCall<
  "DenseElementsAttr::get("
    "RankedTensorType::get({}, getElementTypeOrSelf($0->getType())) , {"
      "$_builder.getZeroAttr(getElementTypeOrSelf($0->getType()))})">;

def : Pat<(TF_PadOp TensorOf<[AnyInteger, AnyFloat]>:$input, $paddings),
          (TF_PadV2Op $input, $paddings,
             (TF_ConstOp (ZeroDenseElementsAttr $input)))>;

