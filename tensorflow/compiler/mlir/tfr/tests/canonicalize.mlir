// RUN: tfr-opt %s -canonicalize -verify-diagnostics -split-input-file | FileCheck %s

// Tests for ops with canonicalization patterns.

// CHECK-LABEL: get_real_shape
func @get_real_shape(%arg0: tensor<1x2xf32>) -> tensor<2xindex> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x2xf32>) -> !tfr.tensor
  %1 = tfr.get_shape %0 -> !shape.shape
  %2 = shape.to_extent_tensor %1 : !shape.shape -> tensor<2xindex>
  return %2 : tensor<2xindex>

// CHECK-NEXT: %[[e:.*]] = shape.const_shape [1, 2] : tensor<2xindex>
// CHECK-NEXT: return %[[e]] : tensor<2xindex>
}

// CHECK-LABEL: equal
func @equal() -> (i1, i1, i1, i1) {
  %0 = tfr.constant f32 -> !tfr.attr
  %1 = tfr.constant f32 -> !tfr.attr
  %2 = tfr.constant i32 -> !tfr.attr
  %same_type = tfr.equal %0,%1 -> i1
  %diff_type = tfr.equal %0,%2 -> i1

  %3 = tfr.constant "hello" -> !tfr.attr
  %4 = tfr.constant "hello" -> !tfr.attr
  %5 = tfr.constant "how are you" -> !tfr.attr
  %same_str = tfr.equal %3,%4 -> i1
  %diff_str = tfr.equal %3,%5 -> i1
  return %same_type, %diff_type, %same_str, %diff_str  : i1, i1, i1, i1

// CHECK-DAG: %true = constant true
// CHECK-DAG: %false = constant false
// CHECK-NEXT: return %true, %false, %true, %false : i1, i1, i1, i1
}

// -----

// CHECK-LABEL: constant_tensor_array
func @constant_tensor_array() -> !tfr.tensor {
  %0 = tfr.constant [1, -1, 3] -> !tfr.attr
  %1 = "tfr.constant_tensor"(%0) : (!tfr.attr) -> !tfr.tensor
  return %1 : !tfr.tensor

// CHECK-NEXT: %[[RES:.*]] = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT: "tfr.cast"(%[[RES]]) : (tensor<3xi64>) -> !tfr.tensor
// CHECK-NEXT: return
}

// -----

// CHECK-LABEL: constant_tensor_scalar
func @constant_tensor_scalar() -> !tfr.tensor {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "tfr.constant_tensor"(%0) : (i32) -> !tfr.tensor
  return %1 : !tfr.tensor

// CHECK-NEXT: %[[RES:.*]] = "tf.Const"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: "tfr.cast"(%[[RES]]) : (tensor<i32>) -> !tfr.tensor
// CHECK-NEXT: return
}

// -----

// CHECK-LABEL: quant_raw_data
func @quant_raw_data(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<1x10x!quant.uniform<i8:f32, 0.2:2>> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1:1>>) -> !tfr.tensor
  %1 = tfr.quant_raw_data(%0) : (!tfr.tensor) -> !tfr.tensor
  %2 = tfr.call @tf__risc(%1) : (!tfr.tensor) -> !tfr.tensor
  %3 = "tfr.cast"(%2) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 0.3:3>>
  %4 = "tfr.cast"(%3) : (tensor<1x10x!quant.uniform<i8:f32, 0.3:3>>) -> !tfr.tensor
  %5 = tfr.quant_raw_data(%4) : (!tfr.tensor) -> !tfr.tensor
  %6 = tfr.call @tf__risc(%5) : (!tfr.tensor) -> !tfr.tensor
  %7 = "tfr.cast"(%6) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 0.2:2>>
  return %7 : tensor<1x10x!quant.uniform<i8:f32, 0.2:2>>
// CHECK: %[[c0:.*]] = "tfr.cast"(%[[arg0:.*]]) : (tensor<1x10x!quant.uniform<i8:f32, 1.000000e-01:1>>) -> !tfr.tensor
// CHECK: %[[c1:.*]] = tfr.call @tf__risc(%[[c0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK: %[[c2:.*]] = tfr.call @tf__risc(%[[c1]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK: %[[c3:.*]] = "tfr.cast"(%[[c2]]) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 2.000000e-01:2>>
// CHECK: return %[[c3]]
}

// -----

// CHECK-LABEL: quant_qparam
func @quant_qparam(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1:42>>) -> (tensor<f32>, tensor<i32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1:42>>) -> !tfr.tensor
  %scale, %zp = tfr.quant_qparam(%0) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
  %1 = "tfr.cast"(%scale) : (!tfr.tensor) -> tensor<f32>
  %2 = "tfr.cast"(%zp) : (!tfr.tensor) -> tensor<i32>
  return %1, %2 : tensor<f32>, tensor<i32>

// CHECK: %[[scale:.*]] = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>}
// CHECK: %[[zp:.*]] = "tf.Const"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
// CHECK: return %[[scale]], %[[zp]]
}

// -----

// CHECK-LABEL: build_const_list
func @build_const_list() -> !tfr.attr {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "std.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  return %2 : !tfr.attr

// CHECK-NEXT: %[[c:.*]] = tfr.constant [42 : i32, 41 : i32] -> !tfr.attr
// CHECK-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: build_high_dim_const_list
func @build_high_dim_const_list() -> !tfr.attr {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "std.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %3 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %4 = "tfr.build_list"(%2, %3) : (!tfr.attr, !tfr.attr) -> !tfr.attr
  return %4 : !tfr.attr

// CHECK-NEXT: %[[c:.*]] = tfr.constant {{\[}}[42 : i32, 41 : i32], [42 : i32, 41 : i32]] -> !tfr.attr
// CHECK-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: get_length
func @get_length(%arg0: !tfr.tensor<A>, %arg1: !tfr.tensor<B>) -> index {
  %0 = "tfr.build_list"(%arg0, %arg1) : (!tfr.tensor<A>, !tfr.tensor<B>) -> !tfr.tensor_list
  %1 = "tfr.get_length"(%0) : (!tfr.tensor_list) -> index
  return %1 : index

// CHECK-NEXT: %[[c:.*]] = constant 2 : index
// CHECK-NEXT: return %[[c]] : index
}
