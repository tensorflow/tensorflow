// RUN: tfr-opt %s -tfr-decompose -verify-diagnostics | FileCheck %s

// Definitions for ops that are being used in the tests.
// ex) tf.MyOp refers to tfr.func @tf__my_op

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
  %cst_1 = arith.constant 1 : i64
  %stride = "tfr.build_list"(%cst_1, %stride_w, %stride_h, %cst_1) : (i64, i64, i64, i64) -> !tfr.attr
  %filter = tfr.constant [1, 2, 2, 1] -> !tfr.attr
  %padding = tfr.constant "VALID" -> !tfr.attr
  %explicit_paddings = tfr.constant [] -> !tfr.attr
  %data_format = tfr.constant "NHWC" -> !tfr.attr
  %MaxPool = tfr.call @tf__max_pool(%input_, %stride, %filter, %padding, %explicit_paddings, %data_format) : (!tfr.tensor, !tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr) -> (!tfr.tensor)
  tfr.return %MaxPool : !tfr.tensor
// CHECK: tf__max_pool
}

// CHECK-LABEL: @tf__cast_float
tfr.func @tf__cast_float(%input_: !tfr.tensor, %out_type: !tfr.attr{tfr.name="out_type"}) -> (!tfr.tensor) {
  %false = arith.constant false
  %cast = tfr.call @tf__cast(%input_, %out_type, %false) : (!tfr.tensor, !tfr.attr, i1) -> (!tfr.tensor)
  tfr.return %cast : !tfr.tensor
}

// end op definitions

// CHECK-LABEL: decompose_tf_no_op
func @decompose_tf_no_op(%arg0: tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string> {
  %0 = "tf.FakeNoOp"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string>
  return %0 : tensor<1x2x3x4x!tf_type.string>

// CHECK-NEXT: return %arg0
}

// CHECK-LABEL: decompose_tf_intermediate
func @decompose_tf_intermediate(%arg0: tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string> {
  %0 = "tf.Intermediate"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string>
  return %0 : tensor<1x2x3x4x!tf_type.string>

// CHECK-NEXT: %[[casted:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> !tfr.tensor
// CHECK-NEXT: %[[id:.*]] = tfr.call @tf__risc(%[[casted]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id]]) : (!tfr.tensor) -> tensor<1x2x3x4x!tf_type.string>
// CHECK-NEXT: return %[[back]]
}

// CHECK-LABEL: decompose_fused_n_default
func @decompose_fused_n_default(%arg0: tensor<1x2x3x4x!tf_type.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) : (tensor<1x2x3x4x!tf_type.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf_type.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in2:.*]] = "tfr.cast"(%arg2) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[id2:.*]] = tfr.call @tf__risc(%[[in2]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id2]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK-LABEL: decompose_fused_n
func @decompose_fused_n(%arg0: tensor<1x2x3x4x!tf_type.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index} : (tensor<1x2x3x4x!tf_type.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf_type.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in1:.*]] = "tfr.cast"(%arg1) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[id1:.*]] = tfr.call @tf__risc(%[[in1]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id1]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK-LABEL: attribute_propagate_direct
func @attribute_propagate_direct(%arg0: tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string> {
  %0 = "tf.Intermediate"(%arg0) {_tpu_replicate, device="hello"} : (tensor<1x2x3x4x!tf_type.string>) -> tensor<1x2x3x4x!tf_type.string>
  return %0 : tensor<1x2x3x4x!tf_type.string>

// CHECK-NEXT: %[[casted:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> !tfr.tensor
// CHECK-NEXT: %[[id:.*]] = tfr.call @tf__risc(%[[casted]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id]]) : (!tfr.tensor) -> tensor<1x2x3x4x!tf_type.string>
// CHECK-NEXT: return %[[back]]
}

// CHECK-LABEL: attribute_propagate
func @attribute_propagate(%arg0: tensor<1x2x3x4x!tf_type.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index, _tpu_replicate, device="hello"} : (tensor<1x2x3x4x!tf_type.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf_type.string>, tensor<f32>)
  return %0#1 : tensor<f32>

// CHECK-NEXT: %[[in0:.*]] = "tfr.cast"(%arg0) : (tensor<1x2x3x4x!tf_type.string>) -> !tfr.tensor
// CHECK-NEXT: %[[in1:.*]] = "tfr.cast"(%arg1) : (tensor<f32>) -> !tfr.tensor
// CHECK-NEXT: %[[id0:.*]] = tfr.call @tf__risc(%[[in0]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[id1:.*]] = tfr.call @tf__risc(%[[in1]]) {_tpu_replicate, device = "hello"}
// CHECK-NEXT: %[[back:.*]] = "tfr.cast"(%[[id1]]) : (!tfr.tensor) -> tensor<f32>
// CHECK-NEXT: return %[[back]] : tensor<f32>
}

// CHECK: attribute_cast
func @attribute_cast(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x4x4x1xf32>) -> !tfr.tensor
  %stride_i32 = arith.constant 2 : i32
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
func @denied_attribute(%arg0: tensor<1x2x3x4x!tf_type.string>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{Denied unregistered attribute was found: denied_attr}}
  %0:2 = "tf.FusedN"(%arg0, %arg1, %arg2) {A=0:index, denied_attr} : (tensor<1x2x3x4x!tf_type.string>, tensor<f32>, tensor<f32>) -> (tensor<1x2x3x4x!tf_type.string>, tensor<f32>)
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
// CHECK-LABEL: decompose_quant_act_range
func @decompose_quant_act_range() -> !tfr.tensor_list {
  %scale = arith.constant 0.1 : f32
  %zp = arith.constant 42 : i64
  %none_attr = tfr.constant "NONE" -> !tfr.attr
  %relu_attr = tfr.constant "RELU" -> !tfr.attr
  %relu6_attr = tfr.constant "RELU6" -> !tfr.attr
  %reluN1_1_attr = tfr.constant "RELU_N1_TO_1" -> !tfr.attr
  %none:2 = "tfr.quant_act_range"(%none_attr, %scale, %zp) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
  %relu:2 = "tfr.quant_act_range"(%relu_attr, %scale, %zp) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
  %relu6:2 = "tfr.quant_act_range"(%relu6_attr, %scale, %zp) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
  %reluN1_1:2 = "tfr.quant_act_range"(%reluN1_1_attr, %scale, %zp) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
  %result = "tfr.build_list"(
    %none#0, %none#1, %relu#0, %relu#1,
    %relu6#0, %relu6#1, %reluN1_1#0, %reluN1_1#1) : (
      !tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor,
      !tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  return %result : !tfr.tensor_list
// CHECK-DAG: %[[N_128:.*]] = arith.constant -128 : i32
// CHECK-DAG: %[[N32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[N42:.*]] = arith.constant 42 : i32
// CHECK-DAG: %[[N52:.*]] = arith.constant 52 : i32
// CHECK-DAG: %[[N102:.*]] = arith.constant 102 : i32
// CHECK-DAG: %[[N127:.*]] = arith.constant 127 : i32
// CHECK-NEXT: %[[none_min:.*]] = "tfr.constant_tensor"(%[[N_128]])
// CHECK-NEXT: %[[none_max:.*]] = "tfr.constant_tensor"(%[[N127]])
// CHECK-NEXT: %[[relu_min:.*]] = "tfr.constant_tensor"(%[[N42]])
// CHECK-NEXT: %[[relu_max:.*]] = "tfr.constant_tensor"(%[[N127]])
// CHECK-NEXT: %[[relu6_min:.*]] = "tfr.constant_tensor"(%[[N42]])
// CHECK-NEXT: %[[relu6_max:.*]] = "tfr.constant_tensor"(%[[N102]])
// CHECK-NEXT: %[[reluN1_1_min:.*]] = "tfr.constant_tensor"(%[[N32]])
// CHECK-NEXT: %[[reluN1_1_max:.*]] = "tfr.constant_tensor"(%[[N52]])
// CHECK-NEXT: %[[result:.*]] = "tfr.build_list"(%[[none_min]], %[[none_max]], %[[relu_min]], %[[relu_max]],
// CHECK-SAME: %[[relu6_min]], %[[relu6_max]], %[[reluN1_1_min]], %[[reluN1_1_max]]
// CHECK-NEXT: return %[[result]]
}

// CHECK-LABEL: decompose_quant_act_range_invalid
func @decompose_quant_act_range_invalid() -> (!tfr.tensor, !tfr.tensor) {
  %scale = arith.constant 0.1 : f32
  %zp = arith.constant 42 : i64
  %elu_attr = tfr.constant "ELU" -> !tfr.attr
  %min, %max = "tfr.quant_act_range"(%elu_attr, %scale, %zp) : (!tfr.attr, f32, i64) -> (!tfr.tensor, !tfr.tensor)
  return %min, %max : !tfr.tensor, !tfr.tensor
// CHECK: %[[elu_attr:.*]] = tfr.constant "ELU" -> !tfr.attr
// CHECK: %[[min:.*]], %[[max:.*]] = tfr.quant_act_range(%[[elu_attr]]
// CHECK: return %[[min]], %[[max]]
}

// CHECK-LABEL: decompose_quant_scale_factor
func @decompose_quant_scale_factor() -> (!tfr.tensor, !tfr.tensor) {
  %output_scale = arith.constant 0.1 : f32
  %input_scale = arith.constant 0.25 : f32
  %filter_scale = arith.constant 0.4 : f32
  %input_scale_tensor = "tfr.constant_tensor"(%input_scale) : (f32) -> !tfr.tensor
  %filter_scale_tensor = "tfr.constant_tensor"(%filter_scale) : (f32) -> !tfr.tensor
  %list = "tfr.build_list"(%input_scale_tensor, %filter_scale_tensor) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  %out = "tfr.quant_scale_factor"(%output_scale, %list) : (f32, !tfr.tensor_list) -> !tfr.tensor
  %perchannel_scale = arith.constant dense<[0.4, 4.0]> : tensor<2xf32>
  %perchannel_scale_tensor = "tfr.cast"(%perchannel_scale) : (tensor<2xf32>) -> !tfr.tensor
  %list2 = "tfr.build_list"(%input_scale_tensor, %perchannel_scale_tensor) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  %perchannel = "tfr.quant_scale_factor"(%output_scale, %list2) : (f32, !tfr.tensor_list) -> !tfr.tensor
  return %out, %perchannel : !tfr.tensor, !tfr.tensor
// CHECK-DAG: %[[scale_factors:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 1.000000e+01]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[scale_factor:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[cast:.*]] = "tfr.cast"(%[[scale_factor]]) : (tensor<f32>) -> !tfr.tensor
// CHECK: %[[cast_perchannel:.*]] = "tfr.cast"(%[[scale_factors]]) : (tensor<2xf32>) -> !tfr.tensor
// CHECK: return %[[cast]], %[[cast_perchannel]] : !tfr.tensor, !tfr.tensor
}

// CHECK-LABEL: decompose_quant_scale_factor_invalid
func @decompose_quant_scale_factor_invalid() -> !tfr.tensor {
  %output_scale = arith.constant 0.1 : f32
  %input_scale = arith.constant 0.25 : f32
  %filter_scale = arith.constant 0.4 : f32
  %input_scale_tensor = "tfr.constant_tensor"(%input_scale) : (f32) -> !tfr.tensor
  %filter_scale_tensor = "tfr.constant_tensor"(%filter_scale) : (f32) -> !tfr.tensor
  %list = "tfr.build_list"(%input_scale_tensor, %filter_scale_tensor, %input_scale_tensor) : (!tfr.tensor, !tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
  %out = "tfr.quant_scale_factor"(%output_scale, %list) : (f32, !tfr.tensor_list) -> !tfr.tensor
  return %out : !tfr.tensor
// CHECK-DAG: %[[cst_0:.*]] = arith.constant 1.000000e-01 : f32
// CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() {value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[cst_2:.*]] = "tf.Const"() {value = dense<4.000000e-01> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[tfrcast0:.*]] = "tfr.cast"(%[[cst_1]]) : (tensor<f32>) -> !tfr.tensor
// CHECK: %[[tfrcast1:.*]] = "tfr.cast"(%[[cst_2]]) : (tensor<f32>) -> !tfr.tensor
// CHECK: %[[list:.*]] = "tfr.build_list"(%[[tfrcast0]], %[[tfrcast1]], %[[tfrcast0]]) : (!tfr.tensor, !tfr.tensor, !tfr.tensor) -> !tfr.tensor_list
// CHECK: %[[qsf:.*]] = tfr.quant_scale_factor(%[[cst_0]], %[[list]]) : (f32, !tfr.tensor_list) -> !tfr.tensor
// CHECK: return %[[qsf]] : !tfr.tensor
}

// CHECK-LABEL: decompose_quant_rescale
func @decompose_quant_rescale(%arg0: tensor<2xi32>) -> !tfr.tensor {
  %zp = arith.constant 67 : i64
  %cst = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %scale_factor = "tfr.cast"(%cst) : (tensor<f32>) -> !tfr.tensor
  %input = "tfr.cast"(%arg0) : (tensor<2xi32>) -> !tfr.tensor
  %rescaled = "tfr.quant_rescale"(%input, %scale_factor, %zp) : (!tfr.tensor, !tfr.tensor, i64) -> !tfr.tensor
  return %rescaled : !tfr.tensor

// CHECK-DAG: %[[scale_cst:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %false = arith.constant false
// CHECK-DAG: %[[zp_cst:.*]] = "tf.Const"() {value = dense<67> : tensor<i64>} : () -> tensor<i64>
// CHECK: %[[zp:.*]] = "tfr.cast"(%[[zp_cst]]) : (tensor<i64>) -> !tfr.tensor
// CHECK: %[[scale:.*]] = "tfr.cast"(%[[scale_cst]]) : (tensor<f32>) -> !tfr.tensor
// CHECK: %[[input:.*]] = "tfr.cast"(%arg0) : (tensor<2xi32>) -> !tfr.tensor
// CHECK-DAG: %[[f32:.*]] = tfr.constant f32 -> !tfr.attr
// CHECK-DAG: %[[i32:.*]] = tfr.constant i32 -> !tfr.attr
// CHECK: %[[cast:.*]] = tfr.call @tf__cast(%[[input]], %[[f32]], %false) : (!tfr.tensor, !tfr.attr, i1) -> !tfr.tensor
// CHECK: %[[rescaled:.*]] = tfr.call @tf__mul(%[[cast]], %[[scale]]) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
// CHECK: %[[rounded:.*]] = tfr.call @tf__round(%[[rescaled]]) : (!tfr.tensor) -> !tfr.tensor
// CHECK: %[[zp_cast:.*]] = tfr.call @tf__cast(%[[zp]], %[[f32]], %false) : (!tfr.tensor, !tfr.attr, i1) -> !tfr.tensor
// CHECK: %[[recentered:.*]] = tfr.call @tf__add(%[[rounded]], %[[zp_cast]]) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
// CHECK: %[[cast_i32:.*]] = tfr.call @tf__cast(%[[recentered]], %[[i32]], %false) : (!tfr.tensor, !tfr.attr, i1) -> !tfr.tensor
// CHECK: return %[[cast_i32]] : !tfr.tensor
}

// CHECK-LABEL: decompose_output_type
func @decompose_output_type(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  %0 = "tf.CastFloat"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
// CHECK: %[[i32:.*]] = tfr.constant i32 -> !tfr.attr
// CHECK: tfr.call @tf__cast(%[[casted_arg:.*]], %[[i32]], %false) : (!tfr.tensor, !tfr.attr, i1) -> !tfr.tensor
}
