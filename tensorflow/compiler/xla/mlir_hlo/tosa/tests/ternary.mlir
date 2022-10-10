// RUN: mhlo-tosa-opt %s --tosa-legalize-mhlo | FileCheck %s

// CHECK-LABEL: @select
func.func @select(%arg0 : tensor<10xi1>, %arg1 : tensor<10xf32>, %arg2 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.select
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<10xi1>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
