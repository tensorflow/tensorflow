// RUN: tfr-opt %s -canonicalize="test-convergence" -verify-diagnostics -split-input-file | FileCheck %s

// Tests for ops with canonicalization patterns.

// CHECK-LABEL: get_real_shape
func.func @get_real_shape(%arg0: tensor<1x2xf32>) -> tensor<2xindex> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x2xf32>) -> !tfr.tensor
  %1 = tfr.get_shape %0 -> !shape.shape
  %2 = shape.to_extent_tensor %1 : !shape.shape -> tensor<2xindex>
  func.return %2 : tensor<2xindex>

// CHECK-NEXT: %[[e:.*]] = shape.const_shape [1, 2] : tensor<2xindex>
// CHECK-NEXT: return %[[e]] : tensor<2xindex>
}

// CHECK-LABEL: equal
func.func @equal() -> (i1, i1, i1, i1) {
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
  func.return %same_type, %diff_type, %same_str, %diff_str  : i1, i1, i1, i1

// CHECK-DAG: %true = arith.constant true
// CHECK-DAG: %false = arith.constant false
// CHECK-NEXT: return %true, %false, %true, %false : i1, i1, i1, i1
}

// -----

// CHECK-LABEL: constant_tensor_array
func.func @constant_tensor_array() -> !tfr.tensor {
  %0 = tfr.constant [1, -1, 3] -> !tfr.attr
  %1 = "tfr.constant_tensor"(%0) : (!tfr.attr) -> !tfr.tensor
  func.return %1 : !tfr.tensor

// CHECK-NEXT: %[[RES:.*]] = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK-NEXT: "tfr.cast"(%[[RES]]) : (tensor<3xi64>) -> !tfr.tensor
// CHECK-NEXT: return
}

// -----

// CHECK-LABEL: constant_tensor_scalar
func.func @constant_tensor_scalar() -> !tfr.tensor {
  %0 = "arith.constant"() {value = 42 : i32} : () -> i32
  %1 = "tfr.constant_tensor"(%0) : (i32) -> !tfr.tensor
  func.return %1 : !tfr.tensor

// CHECK-NEXT: %[[RES:.*]] = "tf.Const"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: "tfr.cast"(%[[RES]]) : (tensor<i32>) -> !tfr.tensor
// CHECK-NEXT: return
}

// -----

// CHECK-LABEL: quant_raw_data
func.func @quant_raw_data(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<1x10x!quant.uniform<i8:f32, 0.2:2>> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1:1>>) -> !tfr.tensor
  %1 = tfr.quant_raw_data(%0) : (!tfr.tensor) -> !tfr.tensor
  %2 = tfr.call @tf__risc(%1) : (!tfr.tensor) -> !tfr.tensor
  %3 = "tfr.cast"(%2) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 0.3:3>>
  %4 = "tfr.cast"(%3) : (tensor<1x10x!quant.uniform<i8:f32, 0.3:3>>) -> !tfr.tensor
  %5 = tfr.quant_raw_data(%4) : (!tfr.tensor) -> !tfr.tensor
  %6 = tfr.call @tf__risc(%5) : (!tfr.tensor) -> !tfr.tensor
  %7 = "tfr.cast"(%6) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 0.2:2>>
  func.return %7 : tensor<1x10x!quant.uniform<i8:f32, 0.2:2>>
// CHECK: %[[c0:.*]] = "tfr.cast"(%[[arg0:.*]]) : (tensor<1x10x!quant.uniform<i8:f32, 1.000000e-01:1>>) -> !tfr.tensor
// CHECK: %[[c1:.*]] = tfr.call @tf__risc(%[[c0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK: %[[c2:.*]] = tfr.call @tf__risc(%[[c1]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK: %[[c3:.*]] = "tfr.cast"(%[[c2]]) : (!tfr.tensor) -> tensor<1x10x!quant.uniform<i8:f32, 2.000000e-01:2>>
// CHECK: return %[[c3]]
}

// -----

// CHECK-LABEL:  quant_raw_data_with_list
func.func @quant_raw_data_with_list(%arg0: !tfr.tensor, %arg1: !tfr.tensor) -> !tfr.tensor {
  %cst_1 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %1 = "tfr.cast"(%arg0) : (!tfr.tensor) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0078420601785182952:-1>>
  %2 = "tfr.cast"(%arg1) : (!tfr.tensor) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0078420601785182952:-1>>
  %3 = "tfr.cast"(%2) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0078420601785182952:-1>>) -> !tfr.tensor
  %4 = "tfr.cast"(%1) : (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.0078420601785182952:-1>>) -> !tfr.tensor
  %5 = "tfr.build_list"(%3, %4) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  %6 = tfr.quant_raw_data(%5) : (!tfr.tensor_list) -> !tfr.tensor_list
  %7 = "tfr.cast"(%cst_1) : (tensor<i64>) -> !tfr.tensor
  %8 = tfr.call @tf__concat(%7, %6) : (!tfr.tensor, !tfr.tensor_list) -> !tfr.tensor
  func.return %8 : !tfr.tensor

// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
// CHECK: %[[BUILD_LIST_0:.*]] = "tfr.build_list"(%arg1, %arg0) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
// CHECK: %[[CAST_0:.*]] = "tfr.cast"(%[[CONST_0]]) : (tensor<i64>) -> !tfr.tensor
// CHECK: %[[CONCAT_O:.*]] = tfr.call @tf__concat(%[[CAST_0]], %[[BUILD_LIST_0]]) : (!tfr.tensor, !tfr.tensor_list) -> !tfr.tensor
// CHECK: return %[[CONCAT_O]] : !tfr.tensor
}

// -----

// CHECK-LABEL:  cast_with_unranked_quant
func.func @cast_with_unranked_quant(%arg0: tensor<*xi8>, %arg1: tensor<*xi8>) -> tensor<*xf32> {
  %0 = "tf.MaximumFloat"(%arg0, %arg1) : (tensor<*xi8>, tensor<*xi8>) -> tensor<*xi8>
  %1 = "tfr.cast"(%0) : (tensor<*xi8>) -> !tfr.tensor
  %2 = "tfr.cast"(%1) : (!tfr.tensor) -> tensor<*x!quant.uniform<i8:f32, 0.0065901698544621468:-19>>
  %3 = "tf.DequantizeFloat"(%2) : (tensor<*x!quant.uniform<i8:f32, 0.0065901698544621468:-19>>) -> tensor<*xf32>
  func.return %3 : tensor<*xf32>
  // The cast ops should not be removed in this case or it will result in an
  // invalid DequantizeFloat op as following:
  // %0 = "tf.MaximumFloat"(%arg0, %arg1) : (tensor<*xi8>, tensor<*xi8>) -> tensor<*xi8>
  // %1 = "tf.DequantizeFloat"(%0) : (tensor<*xi8>) -> tensor<*xf32>
// CHECK: %[[MAXIMUMFLOAT_0:.*]] = "tf.MaximumFloat"(%arg0, %arg1) : (tensor<*xi8>, tensor<*xi8>) -> tensor<*xi8>
// CHECK: %[[CAST_0:.*]] = "tfr.cast"(%[[MAXIMUMFLOAT_0]]) : (tensor<*xi8>) -> !tfr.tensor
// CHECK: %[[CAST_1:.*]] = "tfr.cast"(%[[CAST_0]]) : (!tfr.tensor) -> tensor<*x!quant.uniform<i8:f32, 0.0065901698544621468:-19>>
// CHECK: %[[DEQUANTIZEFLOAT_0:.*]] = "tf.DequantizeFloat"(%[[CAST_1]]) : (tensor<*x!quant.uniform<i8:f32, 0.0065901698544621468:-19>>) -> tensor<*xf32>
// CHECK: return %[[DEQUANTIZEFLOAT_0]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: quant_qparam
func.func @quant_qparam(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1:42>>) -> (tensor<f32>, tensor<i32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1:42>>) -> !tfr.tensor
  %scale, %zp = tfr.quant_qparam(%0) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
  %1 = "tfr.cast"(%scale) : (!tfr.tensor) -> tensor<f32>
  %2 = "tfr.cast"(%zp) : (!tfr.tensor) -> tensor<i32>
  func.return %1, %2 : tensor<f32>, tensor<i32>

// CHECK-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<1.000000e-01> : tensor<f32>}
// CHECK-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
// CHECK: return %[[scale]], %[[zp]]
}

// CHECK-LABEL: quant_qparam_per_channel
func.func @quant_qparam_per_channel(%arg0: tensor<1x3x!quant.uniform<i8:f32:1, {0.1:1, 0.2:2, 0.3:3}>>) -> (tensor<3xf32>, tensor<3xi32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<1x3x!quant.uniform<i8:f32:1, {0.1:1, 0.2:2, 0.3:3}>>) -> !tfr.tensor
  %scale, %zp = tfr.quant_qparam(%0) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
  %1 = "tfr.cast"(%scale) : (!tfr.tensor) -> tensor<3xf32>
  %2 = "tfr.cast"(%zp) : (!tfr.tensor) -> tensor<3xi32>
  func.return %1, %2 : tensor<3xf32>, tensor<3xi32>

// CHECK-DAG: %[[scale:.*]] = "tf.Const"() {value = dense<[1.000000e-01, 2.000000e-01, 3.000000e-01]> : tensor<3xf32>}
// CHECK-DAG: %[[zp:.*]] = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK: return %[[scale]], %[[zp]]
}

// CHECK-LABEL: quant_qparam_invalid
func.func @quant_qparam_invalid(%arg0: tensor<1x3x!quant.calibrated<f32<-1.0:1.0>>>) -> (!tfr.tensor, !tfr.tensor) {
  %0 = "tfr.cast"(%arg0) : (tensor<1x3x!quant.calibrated<f32<-1.0:1.0>>>) -> !tfr.tensor
  %scale, %zp = tfr.quant_qparam(%0) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
  func.return %scale, %zp: !tfr.tensor, !tfr.tensor

// CHECK: %[[scale:.*]], %[[zp:.*]] = tfr.quant_qparam(%[[input:.*]]) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
// CHECK: return %[[scale]], %[[zp]]
}

// -----

// CHECK-LABEL: redundant_cast_with_different_element_type
func.func @redundant_cast_with_different_element_type(%arg0: tensor<*xf32>) -> (tensor<*xi32>, tensor<2xi32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<*xf32>) -> !tfr.tensor
  %1 = "tfr.cast"(%0) : (!tfr.tensor) -> tensor<*xi32>
  %2 = "tfr.cast"(%0) : (!tfr.tensor) -> tensor<2xi32>
  func.return %1, %2 : tensor<*xi32>, tensor<2xi32>

// CHECK: %[[tf_cast_unranked:.*]] = "tf.Cast"(%arg0) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi32>
// CHECK: %[[ensure_shape:.*]] = "tf.EnsureShape"(%arg0) {shape = #tf_type.shape<2>} : (tensor<*xf32>) -> tensor<2xf32>
// CHECK: %[[tf_cast_ranked:.*]] = "tf.Cast"(%[[ensure_shape]]) {Truncate = false} : (tensor<2xf32>) -> tensor<2xi32>
// CHECK: return %[[tf_cast_unranked]], %[[tf_cast_ranked]] :  tensor<*xi32>, tensor<2xi32>
}

// -----

// CHECK-LABEL: redundant_cast_with_quant_type
func.func @redundant_cast_with_quant_type(%arg0: tensor<10x!quant.uniform<i8:f32, 0.0039133410900831223:-128>>) -> (tensor<10xi32>) {
  %0 = "tfr.cast"(%arg0) : (tensor<10x!quant.uniform<i8:f32, 0.0039133410900831223:-128>>) -> !tfr.tensor
  %1 = tfr.quant_raw_data(%0) : (!tfr.tensor) -> !tfr.tensor
  %2 = "tfr.cast"(%1) : (!tfr.tensor) -> tensor<10xi8>
  %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<10xi8>) -> tensor<10xi32>
  func.return %3 : tensor<10xi32>
// CHECK: %[[CAST_0:.*]] = "tfr.cast"(%arg0) : (tensor<10x!quant.uniform<i8:f32, 0.0039133410900831223:-128>>) -> !tfr.tensor
// CHECK: %[[CAST_1:.*]] = "tfr.cast"(%[[CAST_0]]) : (!tfr.tensor) -> tensor<10xi8>
// CHECK: %[[CAST_2:.*]] = "tf.Cast"(%[[CAST_1]]) {Truncate = false} : (tensor<10xi8>) -> tensor<10xi32>
// CHECK: return %[[CAST_2]] : tensor<10xi32>
}

// -----

// CHECK-LABEL: build_const_list
func.func @build_const_list() -> !tfr.attr {
  %0 = "arith.constant"() {value = 42 : i32} : () -> i32
  %1 = "arith.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  func.return %2 : !tfr.attr

// CHECK-NEXT: %[[c:.*]] = tfr.constant [42 : i32, 41 : i32] -> !tfr.attr
// CHECK-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: build_high_dim_const_list
func.func @build_high_dim_const_list() -> !tfr.attr {
  %0 = "arith.constant"() {value = 42 : i32} : () -> i32
  %1 = "arith.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %3 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %4 = "tfr.build_list"(%2, %3) : (!tfr.attr, !tfr.attr) -> !tfr.attr
  func.return %4 : !tfr.attr

// CHECK-NEXT: %[[c:.*]] = tfr.constant {{\[}}[42 : i32, 41 : i32], [42 : i32, 41 : i32]] -> !tfr.attr
// CHECK-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: get_length
func.func @get_length(%arg0: !tfr.tensor<A>, %arg1: !tfr.tensor<B>) -> index {
  %0 = "tfr.build_list"(%arg0, %arg1) : (!tfr.tensor<A>, !tfr.tensor<B>) -> !tfr.tensor_list
  %1 = "tfr.get_length"(%0) : (!tfr.tensor_list) -> index
  func.return %1 : index

// CHECK-NEXT: %[[c:.*]] = arith.constant 2 : index
// CHECK-NEXT: return %[[c]] : index
}
