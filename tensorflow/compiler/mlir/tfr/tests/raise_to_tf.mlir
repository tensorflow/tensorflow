// RUN: tfr-opt %s -tfr-raise-to-tf -verify-diagnostics -split-input-file | FileCheck %s

tfr.func @tf__risc_same_(!tfr.tensor<T>) -> !tfr.tensor<T> attributes {T}
tfr.func @tf__risc_concat_(!tfr.tensor_list<N, T>) -> !tfr.tensor<T> attributes {T, N}
tfr.func @tf__risc_split_(!tfr.tensor<T>, i32 {tfr.name="N"}) -> !tfr.tensor_list<N, T> attributes {T, N}
tfr.func @tf__risc_cast_(!tfr.tensor, !tfr.attr {tfr.name="K"}) -> !tfr.tensor<K> attributes {T, K}
tfr.func @tf__const_(!tfr.attr {tfr.name="value", tfr.type="tensor"},
  !tfr.attr {tfr.name="K",tfr.type="dtype"}) -> !tfr.tensor<K> attributes {T, K}

// CHECK-LABEL: decompose_tf_same
func @decompose_tf_same(%arg0: tensor<1x2x3x4x!tf.string>) -> tensor<1x2x3x4x!tf.string> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
  %1 = tfr.call @tf__risc_same(%0) : (!tfr.tensor) -> !tfr.tensor
  %2 = "tfr.cast"(%1) : (!tfr.tensor) -> tensor<1x2x3x4x!tf.string>
  return %2 : tensor<1x2x3x4x!tf.string>

// CHECK: %[[id:.*]] = "tf.RiscSame"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> tensor<*x!tf.string>
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[id]]) {shape = #tf.shape<1x2x3x4>} : (tensor<*x!tf.string>) -> tensor<1x2x3x4x!tf.string>
// CHECK: return %[[es]] : tensor<1x2x3x4x!tf.string>
}

// CHECK-LABEL: decompose_tf_consecutive
func @decompose_tf_consecutive(%arg0: tensor<1x2x3x4x!tf.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> !tfr.tensor
  %1 = "tfr.cast"(%arg2) : (tensor<f32>) -> !tfr.tensor
  %2 = tfr.call @tf__risc_same(%0) : (!tfr.tensor) -> !tfr.tensor
  %3 = tfr.call @tf__risc_same(%1) : (!tfr.tensor) -> !tfr.tensor
  %4 = "tfr.cast"(%3) : (!tfr.tensor) -> tensor<f32>
  return %4 : tensor<f32>

// CHECK: %[[id0:.*]] = "tf.RiscSame"(%arg0) : (tensor<1x2x3x4x!tf.string>) -> tensor<*x!tf.string>
// CHECK: %[[id2:.*]] = "tf.RiscSame"(%arg2) : (tensor<f32>) -> tensor<*xf32>
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[id2]]) {shape = #tf.shape<>} : (tensor<*xf32>) -> tensor<f32>
// CHECK: return %[[es]] : tensor<f32>
}

// CHECK-LABEL: decompose_tf_concat_n
func @decompose_tf_concat_n(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> {
  %0 = "tfr.cast"(%arg0) : (tensor<f32>) -> !tfr.tensor
  %1 = "tfr.cast"(%arg1) : (tensor<f32>) -> !tfr.tensor
  %2 = "tfr.cast"(%arg2) : (tensor<f32>) -> !tfr.tensor
  %3 = "tfr.build_list"(%0, %1, %2) : (!tfr.tensor, !tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  %concat = tfr.call @tf__risc_concat(%3) : (!tfr.tensor_list) -> !tfr.tensor
  %4 = "tfr.cast"(%concat) : (!tfr.tensor) -> tensor<3xf32>
  return %4 : tensor<3xf32>

// CHECK: %[[concat:.*]] = "tf.RiscConcat"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[concat]]) {shape = #tf.shape<3>} : (tensor<*xf32>) -> tensor<3xf32>
// CHECK: return %[[es]] : tensor<3xf32>
}

// CHECK-LABEL: decompose_tf_split
func @decompose_tf_split(%arg0: tensor<3xf32>) -> (tensor<f32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<3xf32>) -> !tfr.tensor
  %n = std.constant 3: i32
  %split = tfr.call @tf__risc_split(%0, %n) : (!tfr.tensor, i32) -> !tfr.tensor_list
  %i0 = std.constant 0: index
  %s0 = tfr.get_element %split[%i0] : (!tfr.tensor_list, index) -> !tfr.tensor
  %4 = "tfr.cast"(%s0) : (!tfr.tensor) -> tensor<f32>
  return %4 : tensor<f32>

// CHECK: %[[split:.*]]:3 = "tf.RiscSplit"(%arg0) {N = 3 : i32} : (tensor<3xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[split]]#0) {shape = #tf.shape<>} : (tensor<*xf32>) -> tensor<f32>
// CHECK: return %[[es]] : tensor<f32>
}

// CHECK-LABEL: decompose_tf_cast
func @decompose_tf_cast(%arg0: tensor<f32>) -> tensor<i32> {
  %0 = "tfr.cast"(%arg0) : (tensor<f32>) -> !tfr.tensor
  %t = tfr.constant i32 -> !tfr.attr
  %concat = tfr.call @tf__risc_cast(%0, %t) : (!tfr.tensor, !tfr.attr) -> !tfr.tensor
  %4 = "tfr.cast"(%concat) : (!tfr.tensor) -> tensor<i32>
  return %4 : tensor<i32>

// CHECK: %[[tfcast:.*]] = "tf.RiscCast"(%arg0) {K = i32} : (tensor<f32>) -> tensor<*xi32>
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[tfcast]]) {shape = #tf.shape<>} : (tensor<*xi32>) -> tensor<i32>
// CHECK: return %[[es]] : tensor<i32>
}

// CHECK-LABEL: convert_to_scalar_tensor
func @convert_to_scalar_tensor() -> tensor<f32> {
  %0 = constant 3.0: f32
  %t = tfr.constant f32 -> !tfr.attr
  %cst = tfr.call @tf__const(%0, %t) : (f32, !tfr.attr) -> !tfr.tensor
  %4 = "tfr.cast"(%cst) : (!tfr.tensor) -> tensor<f32>
  return %4 : tensor<f32>

// CHECK: %[[cst:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: return %[[cst]] : tensor<f32>
}

// CHECK-LABEL: attribute_propagate
func @attribute_propagate(%arg0: tensor<f32>) -> tensor<i32> {
  %0 = "tfr.cast"(%arg0) : (tensor<f32>) -> !tfr.tensor
  %t = tfr.constant i32 -> !tfr.attr
  %concat = tfr.call @tf__risc_cast(%0, %t) {device = "hello", _tpu_replicate} : (!tfr.tensor, !tfr.attr) -> !tfr.tensor
  %4 = "tfr.cast"(%concat) : (!tfr.tensor) -> tensor<i32>
  return %4 : tensor<i32>

// CHECK: %[[tfcast:.*]] = "tf.RiscCast"(%arg0) {K = i32, _tpu_replicate, device = "hello"} : (tensor<f32>) -> tensor<*xi32>
// CHECK: %[[es:.*]] = "tf.EnsureShape"(%[[tfcast]]) {shape = #tf.shape<>} : (tensor<*xi32>) -> tensor<i32>
// CHECK: return %[[es]] : tensor<i32>
}
