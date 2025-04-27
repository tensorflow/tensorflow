// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: op_constant
func.func @op_constant(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: stablehlo.constant
  // CHECK-NOT: mhlo.constant
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  return %cst : tensor<f32>
}

// -----

// CHECK-LABEL: bounded_dynamic_gather
func.func @bounded_dynamic_gather(%arg0: tensor<16x50xf32>, %arg1: tensor<1x?xi64, #stablehlo.bounds<?,16>>) -> tensor<?x50xf32, #stablehlo.bounds<16, ?>> {
  // CHECK: mhlo.reshape
  // CHECK-NOT: #stablehlo.bounds
  // CHECK-SAME: #mhlo.type_extensions<bounds
  %0 = stablehlo.reshape %arg1 : (tensor<1x?xi64, #stablehlo.bounds<?,16>>) -> tensor<?xi64, #stablehlo.bounds<16>>
  // CHECK: mhlo.gather
  // CHECK-NOT: #stablehlo.bounds
  // CHECK-SAME: #mhlo.type_extensions<bounds
  %1 = "stablehlo.gather"(%arg0, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 50>}> : (tensor<16x50xf32>, tensor<?xi64, #stablehlo.bounds<16>>) -> tensor<?x50xf32, #stablehlo.bounds<16, ?>>
  return %1 : tensor<?x50xf32, #stablehlo.bounds<16, ?>>
}
