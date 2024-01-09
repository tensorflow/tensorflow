// RUN: tac-opt-all-backends -tfl-device-transform-nnapi %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @mean_4d_keepdim(%arg0: tensor<1x48x48x512xf32>) -> tensor<1x1x1x512xf32> {
  %cst = arith.constant dense<[1, 2]> : tensor<2xi32>
  %0 = "tfl.mean"(%arg0, %cst) {keep_dims = true} : (tensor<1x48x48x512xf32>, tensor<2xi32>) -> tensor<1x1x1x512xf32>
  func.return %0 : tensor<1x1x1x512xf32>
}

// CHECK:       func @mean_4d_keepdim([[VAL_0:%.*]]: tensor<1x48x48x512xf32>) -> tensor<1x1x1x512xf32> {
// CHECK:           [[VAL_1:%.*]] = "tfl.average_pool_2d"([[VAL_0]]) {filter_height = 48 : i32, filter_width = 48 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x48x48x512xf32>) -> tensor<1x1x1x512xf32>
// CHECK:           return [[VAL_1]] : tensor<1x1x1x512xf32>
// CHECK:         }

// -----

func.func @mean_4d_no_keepdim(%arg0: tensor<1x48x48x512xf32>) -> tensor<1x512xf32> {
  %cst = arith.constant dense<[1, 2]> : tensor<2xi32>
  %0 = "tfl.mean"(%arg0, %cst) {keep_dims = false} : (tensor<1x48x48x512xf32>, tensor<2xi32>) -> tensor<1x512xf32>
  func.return %0 : tensor<1x512xf32>
}

// CHECK:       func @mean_4d_no_keepdim([[VAL_0:%.*]]: tensor<1x48x48x512xf32>) -> tensor<1x512xf32> {
// CHECK:           [[VAL_1:%.*]] = "tfl.pseudo_const"(){{.*}}dense<[1, 512]> : tensor<2xi32>
// CHECK:           [[VAL_2:%.*]] = "tfl.average_pool_2d"([[VAL_0]]) {filter_height = 48 : i32, filter_width = 48 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x48x48x512xf32>) -> tensor<1x1x1x512xf32>
// CHECK:           [[VAL_3:%.*]] = "tfl.reshape"([[VAL_2]], [[VAL_1]]) : (tensor<1x1x1x512xf32>, tensor<2xi32>) -> tensor<1x512xf32>
// CHECK:           return [[VAL_3]] : tensor<1x512xf32>
// CHECK:         }

// -----

func.func @mean_quant_same_scale(%arg0: tensor<?x7x7x2048x!quant.uniform<i8:f32, 0.6:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.6:-128>> {
   %0 = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
   %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<?x7x7x2048x!quant.uniform<i8:f32, 0.6:-128>>, tensor<2xi32>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.6:-128>>
  func.return %1 : tensor<?x2048x!quant.uniform<i8:f32, 0.6:-128>>
}

// CHECK:   func @mean_quant_same_scale(%[[VAL_0:.*]]: tensor<?x7x7x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>> {
// CHECK:           %[[VAL_1:.*]] = "tfl.pseudo_const"(){{.*}}dense<[-1, 2048]> : tensor<2xi32>
// CHECK:           %[[VAL_2:.*]] = "tfl.average_pool_2d"(%[[VAL_0]]) {filter_height = 7 : i32, filter_width = 7 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x7x7x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>) -> tensor<?x1x1x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>
// CHECK:           %[[VAL_3:.*]] = "tfl.reshape"(%[[VAL_2]], %[[VAL_1]]) : (tensor<?x1x1x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>, tensor<2xi32>) -> tensor<?x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>
// CHECK:           return %[[VAL_3]] : tensor<?x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>
// CHECK:         }

// -----

func.func @mean_quant_different_scales(%arg0: tensor<?x7x7x2048x!quant.uniform<i8:f32, 0.6:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.9:-128>> {
   %0 = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
   %1 = "tfl.mean"(%arg0, %0) {keep_dims = false} : (tensor<?x7x7x2048x!quant.uniform<i8:f32, 0.6:-128>>, tensor<2xi32>) -> tensor<?x2048x!quant.uniform<i8:f32, 0.9:-128>>
  func.return %1 : tensor<?x2048x!quant.uniform<i8:f32, 0.9:-128>>
}

// CHECK:   func @mean_quant_different_scales(%[[VAL_0:.*]]: tensor<?x7x7x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 9.000000e-01:-128>> {
// CHECK:           %[[VAL_1:.*]] = "tfl.pseudo_const"(){{.*}}dense<[-1, 2048]> : tensor<2xi32>
// CHECK:           %[[VAL_2:.*]] = "tfl.average_pool_2d"(%[[VAL_0]]) {filter_height = 7 : i32, filter_width = 7 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x7x7x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>) -> tensor<?x1x1x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>
// CHECK:           %[[VAL_3:.*]] = "tfl.reshape"(%[[VAL_2]], %[[VAL_1]]) : (tensor<?x1x1x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>, tensor<2xi32>) -> tensor<?x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>
// CHECK:           %[[VAL_4:.*]] = "tfl.quantize"(%[[VAL_3]]) {qtype = tensor<?x2048x!quant.uniform<i8:f32, 9.000000e-01:-128>>} : (tensor<?x2048x!quant.uniform<i8:f32, 6.000000e-01:-128>>) -> tensor<?x2048x!quant.uniform<i8:f32, 9.000000e-01:-128>>
// CHECK:           return %[[VAL_4]] : tensor<?x2048x!quant.uniform<i8:f32, 9.000000e-01:-128>>
// CHECK:         }
