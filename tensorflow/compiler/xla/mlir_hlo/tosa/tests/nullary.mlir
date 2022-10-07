// RUN: mhlo-tosa-opt %s --tosa-legalize-mhlo | FileCheck %s

// CHECK-LABEL: @constant
func.func @constant() -> tensor<10xf32> {
  // CHECK: tosa.const
  %0 = mhlo.constant dense<0.000000e+00> : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @constant_f64
func.func @constant_f64() -> tensor<10xf64> {
  // TOSA does not support 64-bit types, so this should not legalize.
  // CHECK: mhlo.constant
  %0 = mhlo.constant dense<0.000000e+00> : tensor<10xf64>
  return %0 : tensor<10xf64>
}
