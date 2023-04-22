/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.td"

class Quantize<string value> : NativeCodeCall<"TFR::Quantize(" # value # ", $0, $1, $_builder)">;

class HasStringAttr<string value> : AttrConstraint<
    CPred<"$_self.cast<StringAttr>().getValue() == \"" # value # "\"">>;

def QuantActRangeNonePattern :
  Pattern<
    (TFR_TFRQuantActRangeOp
     (TFR_ConstOp HasStringAttr<"NONE">:$act), $scale, $zp),
    [(TFR_ConstantTensorOp (ConstantOp ConstantAttr<I32Attr, "-128">)),
     (TFR_ConstantTensorOp (ConstantOp ConstantAttr<I32Attr, "127">))]>;

def QuantActRangeReluPattern :
  Pattern<
    (TFR_TFRQuantActRangeOp
     (TFR_ConstOp HasStringAttr<"RELU">:$act),
     (ConstantLikeMatcher F32Attr:$scale),
     (ConstantLikeMatcher I64Attr:$zp)),
    [(TFR_ConstantTensorOp (ConstantOp (Quantize<"0.0f"> $scale, $zp))),
     (TFR_ConstantTensorOp (ConstantOp ConstantAttr<I32Attr, "127">))]>;

def QuantActRangeRelu6Pattern :
  Pattern<
    (TFR_TFRQuantActRangeOp
     (TFR_ConstOp HasStringAttr<"RELU6">:$act),
     (ConstantLikeMatcher F32Attr:$scale),
     (ConstantLikeMatcher I64Attr:$zp)),
    [(TFR_ConstantTensorOp (ConstantOp (Quantize<"0.0f"> $scale, $zp))),
     (TFR_ConstantTensorOp (ConstantOp (Quantize<"6.0f"> $scale, $zp)))]>;


def QuantActRangeReluN1To1Pattern :
  Pattern<
    (TFR_TFRQuantActRangeOp
     (TFR_ConstOp HasStringAttr<"RELU_N1_TO_1">:$act),
     (ConstantLikeMatcher F32Attr:$scale),
     (ConstantLikeMatcher I64Attr:$zp)),
    [(TFR_ConstantTensorOp (ConstantOp (Quantize<"-1.0f"> $scale, $zp))),
     (TFR_ConstantTensorOp (ConstantOp (Quantize<"1.0f"> $scale, $zp)))]>;
