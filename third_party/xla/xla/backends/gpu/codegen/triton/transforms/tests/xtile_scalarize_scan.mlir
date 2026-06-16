// RUN: xla-opt %s -split-input-file \
// RUN: -xtile-scalarize-scan \
// RUN: -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @tiled_scan_forward
// CHECK-SAME:    %[[ARG0:.*]]: tensor<16x4xf32>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<16xf32>
// CHECK:       %[[output:.*]], %[[final_carry:.*]] = xtile.scan(%[[ARG0]]) inits(%[[ARG1]])
// CHECK-SAME:      dimension = 1 {scan_dim_size = 16 : i64}
// CHECK-SAME:      : (tensor<16x4xf32>), (tensor<16xf32>) -> (tensor<16x4xf32>), (tensor<16xf32>) {
// CHECK:       ^bb0(%arg2: f32, %arg3: f32):
// CHECK:         %[[input:.*]] = tensor.from_elements %arg2 : tensor<f32>
// CHECK:         %[[carry:.*]] = tensor.from_elements %arg3 : tensor<f32>
// CHECK:         %[[add:.*]] = stablehlo.add %[[input]], %[[carry]] : tensor<f32>
// CHECK:         %[[extracted:.*]] = tensor.extract %[[add]][] : tensor<f32>
// CHECK:         xtile.yield %[[extracted]] : f32
// CHECK:       }
// CHECK:       return %[[output]], %[[final_carry]] : tensor<16x4xf32>, tensor<16xf32>

func.func @tiled_scan_forward(%input0: tensor<16x4xf32>, %init0: tensor<16xf32>) -> (tensor<16x4xf32>, tensor<16xf32>) {
  %0, %1 = xtile.scan(%input0) inits(%init0) dimension = 1 {scan_dim_size = 16 : i64} : (tensor<16x4xf32>), (tensor<16xf32>) -> (tensor<16x4xf32>), (tensor<16xf32>) {
  ^bb0(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>):
    %add = stablehlo.add %arg0, %arg1 : tensor<16xf32>
    stablehlo.return %add : tensor<16xf32>
  }
  return %0, %1 : tensor<16x4xf32>, tensor<16xf32>
}
