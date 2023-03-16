// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline=stats-detail-level=1 | \
// RUN: FileCheck %s --check-prefix=CHECK-1

// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline=stats-detail-level=2 | \
// RUN: FileCheck %s --check-prefix=CHECK-2

// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline=stats-detail-level=3 | \
// RUN: FileCheck %s --check-prefix=CHECK-3

func.func @foo(%arg0: tensor<2x4xf32>,
               %arg1: tensor<8x8xf32>,
               %arg2: tensor<128xf32>) -> tensor<4x2xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<2x4xf32>
  %1 = linalg.map { arith.negf }
         ins(%arg0 : tensor<2x4xf32>)
         outs(%0 : tensor<2x4xf32>)
  %3 = tensor.collapse_shape %1 [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
  %4 = tensor.empty() : tensor<8xf32>
  %17 = scf.for %arg13 = %c0 to %c8 step %c32 iter_args(%arg14 = %4)
    -> (tensor<8xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg13] [32] [1] :
        tensor<128xf32> to tensor<32xf32>
    %expanded_17 = tensor.expand_shape %extracted_slice [[0, 1]] :
        tensor<32xf32> into tensor<4x8xf32>
    %reduced_18 = linalg.reduce { arith.addf }
        ins(%expanded_17 : tensor<4x8xf32>)
        outs(%arg14 : tensor<8xf32>) dimensions = [0]
    scf.yield %reduced_18 : tensor<8xf32>
  }
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8xf32>)
      -> tensor<8xf32>
  %6 = linalg.vecmat ins(%3, %arg1 : tensor<8xf32>, tensor<8x8xf32>)
                     outs(%5 : tensor<8xf32>) -> tensor<8xf32>
  %7 = tensor.expand_shape %6 [[0, 1]] : tensor<8xf32> into tensor<8x1xf32>
  %8 = tensor.collapse_shape %7 [[0, 1]] : tensor<8x1xf32> into tensor<8xf32>
  %9 = linalg.matvec ins(%arg1, %8 : tensor<8x8xf32>, tensor<8xf32>)
                     outs(%5 : tensor<8xf32>) -> tensor<8xf32>
  %10 = linalg.map { arith.addf }
         ins(%17, %9 : tensor<8xf32>, tensor<8xf32>)
         outs(%5 : tensor<8xf32>)
  %11 = tensor.expand_shape %10 [[0, 1]] : tensor<8xf32> into tensor<4x2xf32>
  return %11 : tensor<4x2xf32>
}

// CHECK-1:         *** Tileable ops stats (detail level 1) ***
// CHECK-1-DAG:     1x linalg.fill
// CHECK-1-DAG:     2x linalg.map
// CHECK-1-DAG:     1x linalg.matvec
// CHECK-1-DAG:     1x linalg.reduce
// CHECK-1-DAG:     1x linalg.vecmat
// CHECK-1-DAG:     1x tensor.collapse_shape (degenerate)
// CHECK-1-DAG:     1x tensor.collapse_shape (non-degenerate)
// CHECK-1-DAG:     3x tensor.expand_shape

// CHECK-2:         *** Tileable ops stats (detail level 2) ***
// CHECK-2:         1x linalg.fill
// CHECK-2-NEXT:      1. %{{.*}} = linalg.fill ins({{.*}}) outs({{.*}})

// CHECK-3:         *** Tileable ops stats (detail level 3) ***
// CHECK-3:         2x linalg.map
// CHECK-3-DAG:       %{{.*}} = linalg.map { arith.negf } ins({{.*}}) outs({{.*}})
// CHECK-3-NEXT:        Producers:
// CHECK-3-NEXT:          <block argument> {{.*}} index: 0
// CHECK-3-NEXT:          tensor.empty
// CHECK-3-NEXT:        Consumers:
// CHECK-3-NEXT:          tensor.collapse_shape
// CHECK-3-DAG:       %{{.*}} = linalg.map { arith.addf } ins({{.*}}) outs({{.*}})
// CHECK-3-NEXT:        Producers:
// CHECK-3-NEXT:          scf.for
// CHECK-3-NEXT:          linalg.matvec
// CHECK-3-NEXT:          linalg.fill
// CHECK-3-NEXT:        Consumers:
// CHECK-3-NEXT:          tensor.expand_shape
