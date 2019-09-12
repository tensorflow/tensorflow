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
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def CreateTFShapeOp : NativeCodeCall<
    "$_builder.create<TF::ShapeOp>($0->getLoc(), $1, $2)">;

//===----------------------------------------------------------------------===//
// TensorList transformation patterns.
// Note that the pattern below rewrites `TensorList` tensors  (which has type DT_VARIANT)
// into regular tensors. We also assume that each element in the `TensorList` has
// a same constant shape.
//===----------------------------------------------------------------------===//
def ConvertTFTensorListFromTensor : Pat<
  (TF_TensorListFromTensorOp $tensor, $element_shape),
  (replaceWithValue $tensor)>;

def ConvertTFTensorListStack : Pat<
  (TF_TensorListStackOp $input, $element_shape, $num_elements),
  (replaceWithValue $input)>;

def ConvertTFTensorListGetItem : Pat<
  (TF_TensorListGetItemOp $input, $index, $element_shape),
  (TF_GatherOp $input, $index, (NativeCodeCall<"$_builder.getBoolAttr(true)">))>;

// TensorListLength is equivalent to the size of the first dimension of the
// input tensorlist, rewrite it to a combination of Gather and Shape op.
def ConvertTFTensorListLength: Pat<
  (TF_TensorListLengthOp:$old_value $input),
  (TF_GatherOp
    (CreateTFShapeOp $old_value, $input, /*use 32bit*/ConstBoolAttrTrue),
    (ConstantOp ConstantAttr<I32ElementsAttr, "0">), ConstBoolAttrTrue)>;
