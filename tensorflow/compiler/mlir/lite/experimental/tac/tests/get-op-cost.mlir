// RUN: tac-opt-all-backends -tfl-get-op-cost %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @func_0_CPU(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  // CHECK: tac.cost = 7.864320e+05
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<256x32x32x3xf32>, tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  func.return %0 : tensor<256x32x32x3xf32>
}

func.func @func_0_GPU(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32> attributes {tac.device = "GPU", tac.interface_name = "func_0"} {
  // CHECK: tac.cost = 157286.40
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "GPU"} : (tensor<256x32x32x3xf32>, tensor<256x32x32x3xf32>) -> tensor<256x32x32x3xf32>
  func.return %0 : tensor<256x32x32x3xf32>
}

// -----

func.func @func_0_CPU(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  // CHECK: tac.cost = 1.000000e+03
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  func.return %0 : tensor<10x10x10xf32>
}

// -----

func.func @func_0_CPU(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10xf32>) -> tensor<10x10x10xf32> attributes {tac.device = "CPU", tac.interface_name = "func_0"} {
  // CHECK: tac.cost = 1.000000e+03
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  // CHECK: tac.cost = 1.000000e+03
  %1 = "tfl.mul"(%0, %arg1) {fused_activation_function = "RELU", tac.device = "CPU"} : (tensor<10x10x10xf32>, tensor<10xf32>) -> tensor<10x10x10xf32>
  func.return %1 : tensor<10x10x10xf32>
}

// -----

func.func @pack_CPU(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<2x100xf32> attributes {tac.device = "CPU", tac.interface_name = "func_2"} {
  // CHECK: tac.cost = 1.000000e+02
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", values_count = 2 : i32} : (tensor<100xf32>, tensor<100xf32>) -> tensor<2x100xf32>
  func.return %0 : tensor<2x100xf32>
}

func.func @concat_reshape_GPU(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<2x100xf32> attributes {tac.device = "GPU", tac.interface_name = "func_2"} {
  %cst = arith.constant dense<[2, 100]> : tensor<2xi64>
  // CHECK: tac.cost = 4.000000e+01
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<200xf32>
  // CHECK: tac.cost = 4.040000e+01
  %1 = "tfl.reshape"(%0, %cst) {tac.device = "GPU"} : (tensor<200xf32>, tensor<2xi64>) -> tensor<2x100xf32>
  func.return %1 : tensor<2x100xf32>
}

func.func @concat_reshape_CPU(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<2x100xf32> attributes {tac.device = "CPU", tac.interface_name = "func_2"} {
  %cst = arith.constant dense<[2, 100]> : tensor<2xi64>
  // CHECK: tac.cost = 1.000000e+02
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE", tac.device = "CPU"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<200xf32>
  // CHECK: tac.cost = 1.010000e+02
  %1 = "tfl.reshape"(%0, %cst) {tac.device = "CPU"} : (tensor<200xf32>, tensor<2xi64>) -> tensor<2x100xf32>
  func.return %1 : tensor<2x100xf32>
}

// -----

func.func @testConv2DCPU(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: tac.cost = 0x4D5C0000
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6", tac.device = "CPU"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

func.func @testConv2DGPU(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: tac.cost = 0x4C300000
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6", tac.device = "GPU"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2DNoBiasCPU(%arg0: tensor<128x32x32x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: none) -> tensor<128x32x32x64xf32> {
  // CHECK: tac.cost = 0x4DD80000
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6", tac.device = "CPU"} : (tensor<128x32x32x3xf32>, tensor<64x3x3x3xf32>, none) -> tensor<128x32x32x64xf32>
  func.return %0 : tensor<128x32x32x64xf32>
}

func.func @testConv2DNoBiasGPU(%arg0: tensor<128x32x32x3xf32>, %arg1: tensor<64x3x3x3xf32>, %arg2: none) -> tensor<128x32x32x64xf32> {
  // CHECK: tac.cost = 0x4CACCCCD
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6", tac.device = "GPU"} : (tensor<128x32x32x3xf32>, tensor<64x3x3x3xf32>, none) -> tensor<128x32x32x64xf32>
  func.return %0 : tensor<128x32x32x64xf32>
}
