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
