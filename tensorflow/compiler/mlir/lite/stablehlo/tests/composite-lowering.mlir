// RUN: odml-to-stablehlo-opt -composite-lowering -verify-diagnostics %s | FileCheck %s

func.func @hardswish(%arg0: tensor<2xf32>) -> (tensor<*xf32>) {
  %0 = mhlo.composite "aten.hardswish.default" %arg0 {decomposition = @XlaCallModule_aten.hardswish.default.impl_0} : (tensor<2xf32>) -> tensor<2xf32>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}
func.func private @XlaCallModule_aten.hardswish.default.impl_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = mhlo.constant dense<6.000000e+00> : tensor<f32>
  %1 = "mhlo.broadcast_in_dim"(%0) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %2 = mhlo.constant dense<3.40282347E+38> : tensor<f32>
  %3 = "mhlo.broadcast_in_dim"(%2) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %4 = mhlo.constant dense<3.000000e+00> : tensor<f32>
  %5 = "mhlo.broadcast_in_dim"(%4) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %6 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %7 = "mhlo.broadcast_in_dim"(%6) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %8 = mhlo.constant dense<-3.40282347E+38> : tensor<f32>
  %9 = "mhlo.broadcast_in_dim"(%8) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<2xf32>
  %10 = mhlo.add %arg0, %5 : tensor<2xf32>
  %11 = mhlo.clamp %7, %10, %3 : tensor<2xf32>
  %12 = mhlo.clamp %9, %11, %1 : tensor<2xf32>
  %13 = mhlo.multiply %arg0, %12 : tensor<2xf32>
  %14 = mhlo.divide %13, %1 : tensor<2xf32>
  return %14 : tensor<2xf32>
}

// CHECK-LABEL:   func.func @hardswish(
// CHECK-SAME:                         %[[VAL_0:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[VAL_1:.*]] = "tfl.hard_swish"(%[[VAL_0]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           %[[VAL_2:.*]] = "tf.Identity"(%[[VAL_1]]) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
// CHECK:           %[[VAL_3:.*]] = "tf.Identity"(%[[VAL_2]]) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_3]] : tensor<*xf32>
// CHECK:         }