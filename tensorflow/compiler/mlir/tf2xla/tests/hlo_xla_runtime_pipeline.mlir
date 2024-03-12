// RUN: tf-opt -split-input-file -hlo-xla-runtime-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @simple_add(
func.func @simple_add(%arg0: tensor<f64>) -> tensor<f64> {
  // CHECK: arith.addf
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}
