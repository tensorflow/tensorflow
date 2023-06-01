// RUN: tf-opt "-tfxla-verify-legalization=legalize-chlo=true" -verify-diagnostics -split-input-file %s | FileCheck -dump-input=fail %s
// Tests the VerifyTFXLALegalization Pass, that just ensures we don't have
// any illegal ops at the end of the pipeline.

// CHECK-LABEL: allowsMHLO
func.func @allowsMHLO() -> (tensor<8x64x32x4xcomplex<f32>> {mhlo.sharding = ""}) {
  %0 = mhlo.constant dense<(1.000000e+00,-1.000000e+00)> : tensor<128x32x4xcomplex<f32>>
  %1 = mhlo.constant dense<(1.000000e+00,1.000000e+00)> : tensor<8x64x128xcomplex<f32>>
  %2 = "mhlo.einsum"(%1, %0) {einsum_config = "abc,cde->abde"} : (tensor<8x64x128xcomplex<f32>>, tensor<128x32x4xcomplex<f32>>) -> tensor<8x64x32x4xcomplex<f32>>
  return %2 : tensor<8x64x32x4xcomplex<f32>>
}

// -----

func.func @invalid_non_mhlo() -> ( tensor<128x32x4xcomplex<f32>> {mhlo.sharding = ""}) {
  // expected-error @+1 {{Could not legalize op: tf.Const}}
  %cst = "tf.Const"() {value = dense<(1.000000e+00,-1.000000e+00)> : tensor<128x32x4xcomplex<f32>>} : () -> tensor<128x32x4xcomplex<f32>>
  return %cst : tensor<128x32x4xcomplex<f32>>
}

// -----

func.func @invalid_mixed_mhlo() -> (tensor<8x64x128xcomplex<f32>> {mhlo.sharding = ""}) {
  %0 = mhlo.constant dense<(1.000000e+00,-1.000000e+00)> : tensor<128x32x4xcomplex<f32>>
  // expected-error @+1 {{Could not legalize op: tf.Const}}
  %cst_0 = "tf.Const"() {value = dense<(1.000000e+00,1.000000e+00)> : tensor<8x64x128xcomplex<f32>>} : () -> tensor<8x64x128xcomplex<f32>>
  return %cst_0 : tensor<8x64x128xcomplex<f32>>
}

// -----

func.func @fails_chlo(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // expected-error @+1 {{Could not legalize op: chlo.broadcast_add}}
  %0 = "chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}

// -----

func.func @multiple_failures() -> (tensor<8x64x128xcomplex<f32>> {mhlo.sharding = ""}) {
  %0 = mhlo.constant dense<(1.000000e+00,-1.000000e+00)> : tensor<128x32x4xcomplex<f32>>
  // expected-error @+1 {{Could not legalize op: tf.Const}}
  %cst_0 = "tf.Const"() {value = dense<(1.000000e+00,1.000000e+00)> : tensor<8x64x128xcomplex<f32>>} : () -> tensor<8x64x128xcomplex<f32>>
  // expected-error @+1 {{Could not legalize op: tf.XlaEinsum}}
  %1 = "tf.XlaEinsum"(%cst_0, %0) {equation = "abc,cde->abde"} : (tensor<8x64x128xcomplex<f32>>, tensor<128x32x4xcomplex<f32>>) -> tensor<8x64x32x4xcomplex<f32>>
  return %cst_0 : tensor<8x64x128xcomplex<f32>>
}
