// RUN: odml-to-stablehlo-opt %s -build-stablehlo-composite -cse -canonicalize -cse | FileCheck %s --dump-input=fail

module {
  func.func public @build_nested_composite(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %0 = stablehlo.custom_call @mark_tensor(%arg0) {backend_config = "{\22name\22: \22test.outer\22, \22pos\22: 0, \22id\22: \22f8cf5dba52ff4995b9f3810647aa69e2\22, \22is_input\22: true, \22attr\22: null}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.multiply %c_0, %c : tensor<i32>
    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %4 = stablehlo.add %0, %3 : tensor<2x2xf32>
    %5 = stablehlo.custom_call @mark_tensor(%4) {backend_config = "{\22name\22: \22test.inner\22, \22pos\22: 0, \22id\22: \22709cea3466314458963f3262d1deb27e\22, \22is_input\22: true, \22attr\22: null}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %6 = stablehlo.multiply %c, %c : tensor<i32>
    %7 = stablehlo.convert %6 : (tensor<i32>) -> tensor<f32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %9 = stablehlo.add %5, %8 : tensor<2x2xf32>
    %10 = stablehlo.custom_call @mark_tensor(%9) {backend_config = "{\22name\22: \22test.inner\22, \22pos\22: 0, \22id\22: \22709cea3466314458963f3262d1deb27e\22, \22is_input\22: false, \22attr\22: null}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = stablehlo.custom_call @mark_tensor(%10) {backend_config = "{\22name\22: \22test.outer\22, \22pos\22: 0, \22id\22: \22f8cf5dba52ff4995b9f3810647aa69e2\22, \22is_input\22: false, \22attr\22: null}"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %11 : tensor<2x2xf32>
  }
}

// CHECK-LABEL: build_nested_composite
// CHECK: stablehlo.composite "test.outer"
// CHECK: return
// CHECK: func.func private @test.inner.impl
// CHECK: stablehlo.add %arg0,
// CHECK: return
// CHECK: func.func private @test.outer.impl
// CHECK: stablehlo.add %arg0,
// CHECK: stablehlo.composite "test.inner"
// CHECK: return