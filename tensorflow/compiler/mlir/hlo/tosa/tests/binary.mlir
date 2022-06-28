// RUN: mhlo-tosa-opt %s --tosa-legalize-mhlo | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.add
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @maximum
func.func @maximum(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.maximum
  %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @maximum_f64
func.func @maximum_f64(%arg0 : tensor<10xf64>, %arg1 : tensor<10xf64>) -> tensor<10xf64> {
  // CHECK: mhlo.maximum
  %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xf64>
  return %0 : tensor<10xf64>
}

// CHECK-LABEL: @subtract
func.func @subtract(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.sub
  %0 = "mhlo.subtract"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
