/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def ConstDenseElementsI32ZeroAttr
  : NativeCodeCall<"$_builder.create<TFL::ConstOp>($_loc, DenseElementsAttr::get(RankedTensorType::get({}, $_builder.getI32Type()), {0}))">;

def Size1InputRange : NativeCodeCall<
  "SmallVector<Value, 1>{$0}">;

def Size2InputRange : NativeCodeCall<
  "SmallVector<Value, 2>{$0, $1}">;

def Size3InputRange : NativeCodeCall<
  "SmallVector<Value, 3>{$0, $1, $2}">;

def EmptyCustomOptions : NativeCodeCall<
  "TFL::ConstBytesAttr::get($_builder.getContext(), \"\")">;

def CustomOptions : NativeCodeCall<
  "CustomOptions($_builder.getContext(), $0.getDefiningOp()).value()">;

class CreateStringAttr<string values> : NativeCodeCall<
  "$_builder.getStringAttr("# values #")">;

def LegalizeTensorListReserve : Pat<(TF_TensorListReserveOp:$tf_op $shape, $num_elements),
    (TFL_CustomOp (Size2InputRange $shape, $num_elements),
   (CreateStringAttr<"\"TensorListReserve\"">), (CustomOptions $tf_op))>;

def LegalizeTensorListStack : Pat<(TF_TensorListStackOp $input, $shape, $unused_num_elements),
    (TFL_CustomOp (Size2InputRange $input, $shape),
    (CreateStringAttr<"\"TensorListStack\"">), (EmptyCustomOptions))>;

def LegalizeTensorListSetItem : Pat<(TF_TensorListSetItemOp $input, $index, $item, $unused_resize_attr),
    (TFL_CustomOp (Size3InputRange $input, $index, $item),
    (CreateStringAttr<"\"TensorListSetItem\"">), (EmptyCustomOptions))>;

def LegalizeTensorListFromTensor : Pat<(TF_TensorListFromTensorOp $input, $element_shape),
    (TFL_CustomOp (Size2InputRange $input, $element_shape),
    (CreateStringAttr<"\"TensorListFromTensor\"">), (EmptyCustomOptions))>;

def LegalizeTensorListGetItem : Pat<(TF_TensorListGetItemOp $input, $index, $element_shape),
    (TFL_CustomOp (Size3InputRange $input, $index, $element_shape),
    (CreateStringAttr<"\"TensorListGetItem\"">), (EmptyCustomOptions))>;

def LegalizeTensorListEmpty : Pat<(TF_EmptyTensorListOp:$tf_op $element_shape, $unused_max_elements),
    (TFL_CustomOp (Size2InputRange $element_shape, (ConstDenseElementsI32ZeroAttr)),
    (CreateStringAttr<"\"TensorListReserve\"">), (CustomOptions $tf_op))>;

def LegalizeTensorListElementShape : Pat<(TF_TensorListElementShapeOp $input),
    (TFL_CustomOp (Size1InputRange $input),
    (CreateStringAttr<"\"TensorListElementShape\"">), (EmptyCustomOptions))>;

def LegalizeTensorListLength : Pat<(TF_TensorListLengthOp $input),
    (TFL_CustomOp (Size1InputRange $input),
    (CreateStringAttr<"\"TensorListLength\"">), (EmptyCustomOptions))>;