// RUN: tac-opt-all-backends -tfl-compute-cost %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: tac.cost = 7.864320e+05
func @func_0_CPU(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<256x32x32x3xf32>, tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  return %0 : tensor<256x32x32x3xf32>
}

// CHECK: tac.cost = 157286.4
func @func_0_GPU(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32> attributes {tac.device = "GPU", tac.interface_name = "func_0"} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "GPU"} : (tensor<256x32x32x3xf32>, tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  return %0 : tensor<256x32x32x3xf32>
}

// -----

// CHECK: tac.cost = 1.000000e+03
func @func_0_CPU(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

// -----

// CHECK: tac.cost = 2.000000e+02
func @func_0_GPU(%arg0: tensor<10x10x10xf32>, %arg1: tensor<f32>) -> tensor<10x10x10xf32> attributes {tac.device = "GPU", tac.interface_name = "func_0"} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "GPU"} : (tensor<10x10x10xf32>, tensor<f32>) -> tensor<10x10x10xf32>
  return %0 : tensor<10x10x10xf32>
}

// -----

// CHECK: tac.cost = 2.000000e+03
func @func_0_CPU(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  %1 = "tfl.mul"(%0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  return %1 : tensor<10x10x10xf32>
}

// -----

// CHECK: tac.cost = 0x4B673001
func @quantize_ops_CPU_QUANTIZED_INT8(%arg0: tensor<384x512x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.1>>, %arg2: tensor<128x!quant.uniform<i8:f32, 0.2:-128>>, %arg3: tensor<128x!quant.uniform<i8:f32, 0.2:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>  attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "quantize_ops"} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<128x!quant.uniform<i32:f32, 0.7>>, value = dense<0> : tensor<128xi32>} : () -> tensor<128x!quant.uniform<i32:f32, 0.7>>
  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 0.1>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.1>>, tensor<128x!quant.uniform<i32:f32, 0.7>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.9:-4>>
  %2 = "tfl.pseudo_const"() {value = dense<[1, 384, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tfl.reshape"(%1, %2) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<384x128x!quant.uniform<i8:f32, 0.9:-4>>, tensor<3xi32>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.9:-4>>
  %4 = "tfl.mul"(%3, %arg2) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x384x128x!quant.uniform<i8:f32, 0.9:-4>>, tensor<128x!quant.uniform<i8:f32, 0.2:-128>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:3>>
  %5 = "tfl.add"(%4, %arg3) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x384x128x!quant.uniform<i8:f32, 0.3:3>>, tensor<128x!quant.uniform<i8:f32, 0.2:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>
  return %5 : tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>
}
