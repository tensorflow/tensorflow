// RUN: tfr-opt %s -tfr-decompose -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: @tf__fake_no_op
tfr.func @tf__fake_no_op(%arg0: !tfr.tensor) -> !tfr.tensor {
  tfr.return %arg0 : !tfr.tensor

// CHECK-NEXT: tfr.return %arg0 : !tfr.tensor
}

// CHECK-LABEL: @tf__intermediate
tfr.func @tf__intermediate(%arg0: !tfr.tensor) -> !tfr.tensor {
  %0 = tfr.call @tf__risc(%arg0) : (!tfr.tensor) -> !tfr.tensor
  tfr.return %0 : !tfr.tensor

// CHECK-NEXT: %[[id:.*]] = tfr.call @tf__risc(%arg0) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: tfr.return %[[id]] : !tfr.tensor
}

// CHECK-LABEL: @tf__fused_n
tfr.func @tf__fused_n(
    %arg0: !tfr.tensor,
    %arg1: !tfr.tensor_list,
    %arg2: index {tfr.name="A",tfr.default=1:index})
  -> !tfr.tensor_list {
  %0 = tfr.call @tf__intermediate(%arg0) : (!tfr.tensor) -> !tfr.tensor
  %1 = tfr.get_element %arg1[%arg2] : (!tfr.tensor_list, index) -> !tfr.tensor
  %2 = tfr.call @tf__intermediate(%1) : (!tfr.tensor) -> !tfr.tensor
  %3 = "tfr.build_list"(%0, %2) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  tfr.return %3 : !tfr.tensor_list

// CHECK-NEXT: %[[id1:.*]] = tfr.call @tf__intermediate(%arg0) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[ge:.*]] = tfr.get_element %arg1[%arg2] : (!tfr.tensor_list, index) -> !tfr.tensor
// CHECK-NEXT: %[[id2:.*]] = tfr.call @tf__intermediate(%[[ge]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[bl:.*]] = "tfr.build_list"(%[[id1]], %[[id2]]) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
// CHECK-NEXT: tfr.return %[[bl]] : !tfr.tensor_list
}

// CHECK-LABEL: @tf__my_max_pool
tfr.func @tf__my_max_pool(%input_: !tfr.tensor, %stride_w: i64{tfr.name="stride_w"}, %stride_h: i64{tfr.name="stride_h"}) -> (!tfr.tensor) {
  %cst_1 = constant 1 : i64
  %stride = "tfr.build_list"(%cst_1, %stride_w, %stride_h, %cst_1) : (i64, i64, i64, i64) -> !tfr.attr
  %filter = tfr.constant [1, 2, 2, 1] -> !tfr.attr
  %padding = tfr.constant "VALID" -> !tfr.attr
  %explicit_paddings = tfr.constant [] -> !tfr.attr
  %data_format = tfr.constant "NHWC" -> !tfr.attr
  %MaxPool = tfr.call @tf__max_pool(%input_, %stride, %filter, %padding, %explicit_paddings, %data_format) : (!tfr.tensor, !tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr) -> (!tfr.tensor)
  tfr.return %MaxPool : !tfr.tensor
// CHECK: tf__max_pool
}

//------------------------

// CHECK-LABEL: decompose_tf_no_op
func @decompose_tf_no_op(%arg0: tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string> {
  %0 = "tf.FakeNoOp"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string>
  return %0 : tensor<1x2x3x4x!tf.string>

// CHECK-NEXT: return %arg0
}

// CHECK-LABEL: decompose_tf_intermediate
func @decompose_tf_intermediate(%arg0: tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string> {
  %0 = "tf.Intermediate"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string>
  return %0 : tensor<1x2x3x4x!tf.string>

// CHECK-NEXT: %[[casted:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
// CHECK-NEXT: %[[id:.*]] = tfr.call @tf__risc(%[[casted]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id]]) : (!tfr.tensor) -> tensor<1x2x3x4x!tf.string>
// CHECK-NEXT: return %[[back]]
}

// CHECK-LABEL: decompose_fused_n_default
func @decompose_fused_n_default(%arg0: tensor<1x2x3x4x!tf.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) : (tensor<1x2x3x4x!tf.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in2:.*]] = "tfr.cast"(%arg2) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[id2:.*]] = tfr.call @tf__risc(%[[in2]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id2]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK-LABEL: decompose_fused_n
func @decompose_fused_n(%arg0: tensor<1x2x3x4x!tf.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index} : (tensor<1x2x3x4x!tf.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in1:.*]] = "tfr.cast"(%arg1) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[id1:.*]] = tfr.call @tf__risc(%[[in1]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id1]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK-LABEL: attribute_propagate_direct
func @attribute_propagate_direct(%arg0: tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string> {
  %0 = "tf.Intermediate"(%arg0) {_tpu_replicate, device="hello"} : (tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string>
  return %0 : tensor<1x2x3x4x!tf.string>

// CHECK-NEXT: %[[casted:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
// CHECK-NEXT: %[[id:.*]] = tfr.call @tf__risc(%[[casted]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id]]) : (!tfr.tensor) -> tensor<1x2x3x4x!tf.string>
// CHECK-NEXT: return %[[back]]
}

// CHECK-LABEL: attribute_propagate
func @attribute_propagate(%arg0: tensor<1x2x3x4x!tf.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index, _tpu_replicate, device="hello"} : (tensor<1x2x3x4x!tf.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in1:.*]] = "tfr.cast"(%arg1) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[id1:.*]] = tfr.call @tf__risc(%[[in1]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id1]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK: attribute_cast
func @attribute_cast(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x4x4x1xf32>) -> !tfr.tensor
  %stride_i32 = constant 2 : i32
  %1 = tfr.call @tf__my_max_pool(%0, %stride_i32, %stride_i32) : (!tfr.tensor, i32, i32) -> !tfr.tensor
  %2 = "tfr.cast"(%1) : (!tfr.tensor) -> tensor<1x2x2x1xf32>
  return %2 : tensor<1x2x2x1xf32>
// CHECK: tf__max_pool
}

// CHECK-LABEL: no_tf_canonicalization
func @no_tf_canonicalization(%arg0: tensor<8xi1>, %arg1: tensor<8x3xf32>, %arg2: tensor<8x3xf32>) -> tensor<8x3xf32> {
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<8xi1>, tensor<8x3xf32>, tensor<8x3xf32>) -> tensor<8x3xf32>
  return %0: tensor<8x3xf32>

// CHECK:   "tf.Select"
}

// CHECK-LABEL: denied_attribute
func @denied_attribute(%arg0: tensor<1x2x3x4x!tf.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Denied unregistered attribute was found: denied_attr}}
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index, denied_attr} : (tensor<1x2x3x4x!tf.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT:   "tf.FusedN"(%arg0, %arg1, %arg2) {A = 0 : index, denied_attr}
}

// CHECK-LABEL: quantized_tensor
func @quantized_tensor(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.0038396485615521669:-128>>) -> tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>> {
  %0 = "tf.Intermediate"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.0038396485615521669:-128>>) -> tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  return %0 : tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>>

// CHECK: "tfr.cast"(%[[arg0:.*]]) : (tensor<1x10x!quant.uniform<i8:f32, 0.0038396485615521669:-128>>) -> !tfr.tensor
// CHECK: "tfr.cast"(%[[result:.*]]) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 3.906250e-03:-128>>
}
