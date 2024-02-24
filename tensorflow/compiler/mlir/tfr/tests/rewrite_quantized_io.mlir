// RUN: tfr-opt %s -tfr-decompose -tfr-rewrite-quantized-io -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @tf__my_requantize
tfr.func @tf__my_requantize(%input: !tfr.tensor) -> !tfr.tensor {
  %raw_data = tfr.quant_raw_data(%input) : (!tfr.tensor) -> !tfr.tensor
  %scale, %zp = tfr.quant_qparam(%input) : (!tfr.tensor) -> (!tfr.tensor, !tfr.tensor)
  %result = tfr.call @tf__requantize(%raw_data, %scale, %zp) : (!tfr.tensor, !tfr.tensor, !tfr.tensor) -> !tfr.tensor
  tfr.return %result : !tfr.tensor
}

// CHECK-LABEL: @tf__intermediate
tfr.func @tf__intermediate(%arg0: !tfr.tensor) -> !tfr.tensor {
  %0 = tfr.call @tf__risc(%arg0) : (!tfr.tensor) -> !tfr.tensor
  tfr.return %0 : !tfr.tensor
}

// CHECK-LABEL: remove_quantized_io
func.func @remove_quantized_io(
  %arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1:-128>>,
  %arg1: tensor<1x5xf32>) -> (tensor<1x10x!quant.uniform<i8:f32, 0.2:42>>, tensor<1x5xf32>) {
  %0 = "tf.MyRequantize"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1:-128>>) -> tensor<1x10x!quant.uniform<i8:f32, 0.2:42>>
  %1 = "tf.Intermediate"(%arg1) : (tensor<1x5xf32>) -> tensor<1x5xf32>
  func.return %0, %1 : tensor<1x10x!quant.uniform<i8:f32, 0.2:42>>, tensor<1x5xf32>

// CHECK-DAG: %[[scale:.*]] = "tf.Const"() <{value = dense<1.000000e-01> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG: %[[zp:.*]] = "tf.Const"() <{value = dense<-128> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[quant:.*]] = "tfr.cast"(%arg0) : (tensor<1x10xi8>) -> !tfr.tensor
// CHECK: %[[scale_cast:.*]] = "tfr.cast"(%[[scale]])
// CHECK: %[[zp_cast:.*]] = "tfr.cast"(%[[zp]])
// CHECK: %[[requant:.*]] = tfr.call @tf__requantize(%[[quant]], %[[scale_cast]], %[[zp_cast]])
// CHECK: %[[result:.*]] = "tfr.cast"(%[[requant]])
// CHECK-NOT: quant.uniform
// CHECK: return %[[result]], %[[float_resunt:.*]] : tensor<1x10xi8>, tensor<1x5xf32>
}


// CHECK-LABEL: quant_input_multiple_users
// expected-error@+1 {{The argument with type tensor<1x10x!quant.uniform<i8:f32, 1.000000e-01>> should have one user}}
func.func @quant_input_multiple_users(%arg0: tensor<1x10x!quant.uniform<i8:f32, 0.1>>) -> (!tfr.tensor, !tfr.tensor) {
  %0 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1>>) -> !tfr.tensor
  %1 = "tfr.cast"(%arg0) : (tensor<1x10x!quant.uniform<i8:f32, 0.1>>) -> !tfr.tensor
  func.return %0, %1 : !tfr.tensor, !tfr.tensor
}

