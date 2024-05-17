// RUN: odml-converter --outline-composites %s -split-input-file | FileCheck %s

func.func @geluWithCustomCallErf(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
  %2 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
  %3 = stablehlo.multiply %arg0, %2 : tensor<2xf32>
  %4 = stablehlo.multiply %arg0, %1 : tensor<2xf32>
  %5 = stablehlo.custom_call @mhlo.erf(%4) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<2xf32>) -> tensor<2xf32>
  %6 = stablehlo.add %5, %0 : tensor<2xf32>
  %7 = stablehlo.multiply %3, %6 : tensor<2xf32>
  return %7 : tensor<2xf32>
}

// CHECK: func.func private @gelu_decomp_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
// CHECK: %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK: %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
// CHECK: %cst_1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
// CHECK: %0 = stablehlo.multiply %arg0, %cst_1 : tensor<2xf32>
// CHECK: %1 = chlo.erf %0 : tensor<2xf32> -> tensor<2xf32>
// CHECK: %2 = stablehlo.add %1, %cst : tensor<2xf32>
// CHECK: %3 = stablehlo.multiply %arg0, %cst_0 : tensor<2xf32>
// CHECK: %4 = stablehlo.multiply %3, %2 : tensor<2xf32>
// CHECK: return %4 : tensor<2xf32>

// CHECK-LABEL: geluWithCustomCallErf
// CHECK: %0 = stablehlo.composite "odml.internal.gelu" %arg0 {composite_attributes = {approx = false}, decomposition = @gelu_decomp_0} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK: return %0

// -----

func.func @geluWithCHLOErf(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
  %2 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
  %3 = stablehlo.multiply %arg0, %2 : tensor<2xf32>
  %4 = stablehlo.multiply %arg0, %1 : tensor<2xf32>
  %5 = chlo.erf %4 : tensor<2xf32> -> tensor<2xf32>
  %6 = stablehlo.add %5, %0 : tensor<2xf32>
  %7 = stablehlo.multiply %3, %6 : tensor<2xf32>
  return %7 : tensor<2xf32>
}

// CHECK: func.func private @gelu_decomp_0(%arg0: tensor<2xf32>) -> tensor<2xf32>
// CHECK: %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// CHECK: %cst_0 = stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
// CHECK: %cst_1 = stablehlo.constant dense<0.707106769> : tensor<2xf32>
// CHECK: %0 = stablehlo.multiply %arg0, %cst_1 : tensor<2xf32>
// CHECK: %1 = chlo.erf %0 : tensor<2xf32> -> tensor<2xf32>
// CHECK: %2 = stablehlo.add %1, %cst : tensor<2xf32>
// CHECK: %3 = stablehlo.multiply %arg0, %cst_0 : tensor<2xf32>
// CHECK: %4 = stablehlo.multiply %3, %2 : tensor<2xf32>
// CHECK: return %4 : tensor<2xf32>

// CHECK-LABEL: geluWithCHLOErf
// CHECK: %0 = stablehlo.composite "odml.internal.gelu" %arg0 {composite_attributes = {approx = false}, decomposition = @gelu_decomp_0} : (tensor<2xf32>) -> tensor<2xf32>
// CHECK: return %0
