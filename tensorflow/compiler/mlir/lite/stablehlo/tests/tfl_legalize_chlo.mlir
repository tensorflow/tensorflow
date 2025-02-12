// RUN: odml-to-stablehlo-opt %s -tfl-legalize-chlo -split-input-file | FileCheck %s --dump-input=fail

// Just assert that pass is properly registered.
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0: tensor<f32>
}
// CHECK-LABEL: main

// -----

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

// CHECK-LABEL: geluWithCustomCallErf
// CHECK: "tfl.gelu"
// CHECK-NOT: stablehlo
// CHECK-NOT: chlo

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

// CHECK-LABEL: geluWithCHLOErf
// CHECK: "tfl.gelu"
// CHECK-NOT: stablehlo
// CHECK-NOT: chlo
