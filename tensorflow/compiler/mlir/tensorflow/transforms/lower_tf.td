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
// Pad op patterns.
//===----------------------------------------------------------------------===//

def ZeroDenseElementsAttr : NativeCodeCall<
  "DenseElementsAttr::get("
    "RankedTensorType::get({}, getElementTypeOrSelf($0->getType())) , {"
      "$_builder.getZeroAttr(getElementTypeOrSelf($0->getType()))})">;

def : Pat<(TF_PadOp TensorOf<[AnyInteger, AnyFloat]>:$input, $paddings),
          (TF_PadV2Op $input, $paddings,
             (TF_ConstOp (ZeroDenseElementsAttr $input)))>;

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
