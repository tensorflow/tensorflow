// RUN: litert-opt %s --tfl-canonicalize-boundary-value --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @clamp_neg_inf_f32() -> tensor<f32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
// CHECK:           return %[[CONST]] : tensor<f32>

func.func @clamp_neg_inf_f32() -> tensor<f32> {
  %ret = stablehlo.constant dense<0xFF800000> : tensor<f32>
  return %ret : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f32() -> tensor<f32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<3.40282347E+38> : tensor<f32>
// CHECK:           return %[[CONST]] : tensor<f32>
func.func @clamp_pos_inf_f32() -> tensor<f32> {
  %ret = stablehlo.constant dense<0x7F800000> : tensor<f32>
  return %ret : tensor<f32>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f32_tensor() -> tensor<1x4xf32> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<{{\[\[}}3.40282347E+38, 1.000000e+01, 2.000000e+01, -3.40282347E+38]]> : tensor<1x4xf32>
// CHECK:           return %[[CONST]] : tensor<1x4xf32>
func.func @clamp_pos_inf_f32_tensor() -> tensor<1x4xf32> {
  %ret = stablehlo.constant dense<[[0x7F800000, 10.0, 20.0, 0xFF800000]]> : tensor<1x4xf32>
  return %ret : tensor<1x4xf32>
}

// -----

// CHECK-LABEL:   func.func @clamp_neg_inf_f16() -> tensor<f16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-6.550400e+04> : tensor<f16>
// CHECK:           return %[[CONST]] : tensor<f16>
func.func @clamp_neg_inf_f16() -> tensor<f16> {
  %ret = stablehlo.constant dense<0xFC00> : tensor<f16>
  return %ret : tensor<f16>
}
// -----

// CHECK-LABEL:   func.func @clamp_neg_inf_bf16() -> tensor<bf16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<-1.038460e+34> : tensor<bf16>
// CHECK:           return %[[CONST]] : tensor<bf16>
func.func @clamp_neg_inf_bf16() -> tensor<bf16> {
  %ret = stablehlo.constant dense<0xF800> : tensor<bf16>
  return %ret : tensor<bf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16() -> tensor<f16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<6.550400e+04> : tensor<f16>
// CHECK:           return %[[CONST]] : tensor<f16>
func.func @clamp_pos_inf_f16() -> tensor<f16> {
  %ret = stablehlo.constant dense<0x7C00> : tensor<f16>
  return %ret : tensor<f16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_tensor() -> tensor<1x4xf16> {
// CHECK:           %[[CONST:.*]] = stablehlo.constant dense<{{\[\[}}6.550400e+04, 1.000000e+01, 2.000000e+01, -6.550400e+04]]> : tensor<1x4xf16>
// CHECK:           return %[[CONST]] : tensor<1x4xf16>
func.func @clamp_pos_inf_f16_tensor() -> tensor<1x4xf16> {
  %ret = stablehlo.constant dense<[[0x7C00, 10.0, 20.0, 0xFC00]]> : tensor<1x4xf16>
  return %ret : tensor<1x4xf16>
}

// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_tensor_tf_const() -> tensor<3xf16> {
// CHECK:           %[[CONST:.*]] = "tf.Const"() <{value = dense<6.550400e+04> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK:           return %[[CONST]] : tensor<3xf16>
func.func @clamp_pos_inf_f16_tensor_tf_const() -> tensor<3xf16> {
  %ret = "tf.Const"() <{value = dense<0x7C00> : tensor<3xf16>}> : () -> tensor<3xf16>
  return %ret : tensor<3xf16>
}


// -----

// CHECK-LABEL:   func.func @clamp_pos_inf_f16_arith_op() -> f16 {
// CHECK:           %[[CONST:.*]] = arith.constant 6.550400e+04 : f16
// CHECK:           return %[[CONST]] : f16
func.func @clamp_pos_inf_f16_arith_op() -> f16 {
  %ret = arith.constant 0x7C00 : f16
  return %ret : f16
}

