// RUN: xla-opt -split-input-file -hlo-xla-runtime-pipeline %s | FileCheck %s

// CHECK-LABEL: @simple_add(
func.func @simple_add(%arg0: tensor<f64>) -> tensor<f64> attributes {rt.entrypoint} {
  // CHECK: linalg.generic
  // CHECK: addf
  %0 = mhlo.add %arg0, %arg0 : tensor<f64>
  return %0 : tensor<f64>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

// CHECK-LABEL:   func.func @dense_abs_eltwise(
func.func @dense_abs_eltwise(%arg0: tensor<10x20xf32, #CSR>)
    -> tensor<10x20xf32, #CSR> {
  // CHECK: call @sparsePointers0
  // CHECK: call @sparseIndices0
  // CHECK: call @sparseValuesF32
  // CHECK: math.absf
  %0 = mhlo.abs %arg0 : tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}