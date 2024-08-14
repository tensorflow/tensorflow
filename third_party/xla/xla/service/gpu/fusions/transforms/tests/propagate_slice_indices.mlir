// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-propagate-slice-indices | FileCheck %s

module {
  func.func private @add(%arg0: f32, %arg1: f32) -> f32 {
    %sum = arith.addf %arg0, %arg1 : f32
    func.return %sum : f32
  }

  func.func private @tensorarg(%arg0: tensor<43xf32>, %arg1: index) -> f32 {
    %v1 = arith.constant 2.0 : f32
    %v2 = tensor.extract %arg0[%arg1] : tensor<43xf32>
    %sum = func.call @add(%v1, %v2) : (f32, f32) -> f32
    func.return %sum : f32
  }

  func.func @tensorcall(%arg0: tensor<43xf32>, %arg1: index) -> f32 {
    %call = func.call @tensorarg(%arg0, %arg1) : (tensor<43xf32>, index) -> f32
    func.return %call : f32
  }

  func.func @stores(%arg0: tensor<17xf32> {xla.invariant, xla.slice_index = 0},
                    %arg1: tensor<43xf32> {xla.slice_index = 1}) -> tensor<43xf32>
                    attributes { xla.entry } {
    %c17 = arith.constant 17 : index
    %c23 = arith.constant 23 : index
    %cst = arith.constant 3.0 : f32
    %out = tensor.insert %cst into %arg1[%c17] : tensor<43xf32>
    %out2 = tensor.insert %cst into %out[%c23] : tensor<43xf32>
    func.return %out2 : tensor<43xf32>
  }
}

// CHECK-DAG: @add(%{{.*}}: f32, %{{.*}}: f32)
// CHECK-DAG: @tensorarg(%{{.*}}: tensor<43xf32> {xla.invariant, xla.slice_index = 0 : i64}, %{{.*}}: index)
// CHECK-DAG: @tensorcall(%{{.*}}: tensor<43xf32> {xla.invariant, xla.slice_index = 0 : i64}, %{{.*}}: index)
// CHECK-DAG: @stores(%{{.*}}: tensor<17xf32> {xla.invariant, xla.slice_index = 0 : i64}, %{{.*}}: tensor<43xf32> {xla.slice_index = 1 : i64})
