// RUN: xla-opt %s -split-input-file \
// RUN: -xtile-strip-mine-scan \
// RUN: -canonicalize -cse \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func @tiled_scan_reverse
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c16 = arith.constant 16 : index
// CHECK:     %c4 = arith.constant 4 : index
// CHECK:     %[[FOR_RES:.*]] = scf.for %[[IV:.*]] = %c0 to %c16 step %c4 iter_args(%[[ITER_ARG:.*]] = %arg1) -> (tensor<16xf32>) {
// CHECK:       %[[REM:.*]] = arith.subi %c12, %[[IV]] : index
// CHECK:       %[[EXTRACTED:.*]] = xtile.extract %arg0[%c0, %[[REM]]] [16, 4] [1, 1] : memref<16x16xf32> -> tensor<16x4xf32>
// CHECK:       %[[OUTPUT:.*]], %[[CARRY:.*]] = xtile.scan(%[[EXTRACTED]]) inits(%[[ITER_ARG]]) dimension = 1 {is_reverse = true, scan_dim_size = 4 : i64} : (tensor<16x4xf32>), (tensor<16xf32>) -> (tensor<16x4xf32>), (tensor<16xf32>) {
// CHECK:       ^bb0(%[[ARG_A:.*]]: f32, %[[ARG_B:.*]]: f32):
// CHECK:         %[[FROM_A:.*]] = tensor.from_elements %[[ARG_A]] : tensor<f32>
// CHECK:         %[[FROM_B:.*]] = tensor.from_elements %[[ARG_B]] : tensor<f32>
// CHECK:         %[[ADD:.*]] = stablehlo.add %[[FROM_A]], %[[FROM_B]] : tensor<f32>
// CHECK:         %[[EXTRACT:.*]] = tensor.extract %[[ADD]][] : tensor<f32>
// CHECK:         xtile.yield %[[EXTRACT]] : f32
// CHECK:       }
// CHECK:       xtile.insert %[[OUTPUT]] into %arg2[%c0, %[[REM]]] [16, 4] [1, 1] : tensor<16x4xf32> -> memref<16x16xf32>
// CHECK:       scf.yield %[[CARRY]] : tensor<16xf32>
// CHECK:     }
// CHECK:     return %[[FOR_RES]] : tensor<16xf32>
// CHECK:   }

func.func @tiled_scan_reverse(%input0: memref<16x16xf32>, %init0: tensor<16xf32>, %out0: memref<16x16xf32>) -> (tensor<16xf32>) {
  %c0 = arith.constant 0 : index
  %in_tile0 = xtile.extract %input0[%c0, %c0] [16, 4] [1, 1] : memref<16x16xf32> -> tensor<16x4xf32>

  %0, %1 = xtile.scan(%in_tile0) inits(%init0) dimension = 1 {scan_dim_size = 16 : i64, is_reverse = true} : (tensor<16x4xf32>), (tensor<16xf32>) -> (tensor<16x4xf32>), (tensor<16xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %from_elements = tensor.from_elements %arg0 : tensor<f32>
    %from_elements_0 = tensor.from_elements %arg1 : tensor<f32>
    %add1 = stablehlo.add %from_elements, %from_elements_0 : tensor<f32>
    %extracted = tensor.extract %add1[] : tensor<f32>
    xtile.yield %extracted : f32
  }

  xtile.insert %0 into %out0[%c0, %c0] [16, 4] [1, 1] : tensor<16x4xf32> -> memref<16x16xf32>

  return %1 : tensor<16xf32>
}
