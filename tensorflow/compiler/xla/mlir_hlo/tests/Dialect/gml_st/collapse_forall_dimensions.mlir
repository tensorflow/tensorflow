// RUN: mlir-hlo-opt %s --gml-compose-extract-insert-slice | FileCheck %s

func.func @inline_single_iteration_parallel(
    %in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = scf.forall (%arg4, %arg5) = (%c0, %c0) to (%c1, %c1)
        step (%c8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %20 = tensor.extract_slice %out_[%arg4, %arg5] [8, 8] [1, 1]
      : tensor<8x8xf32> to tensor<8x8xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%20 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %out_[%arg4, %arg5] [8, 8] [1, 1]
        : tensor<8x8xf32> into tensor<8x8xf32>
    }
  }
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @inline_single_iteration_parallel
// CHECK-NOT:     scf.forall
// CHECK:         tensor.empty
// CHECK-NEXT:    linalg.fill

// -----

func.func @collapse_one_dim_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = scf.forall (%arg4, %arg5) = (%c0, %c0) to (%c1, %c16)
        step (%c8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %11 = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %out_[%arg4, %arg5] [8, 8] [1, 1]
        : tensor<8x8xf32> into tensor<8x8xf32>
    }
  }
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @collapse_one_dim_parallel
// CHECK:         scf.forall (%[[ARG:.*]]) = (%c0) to (%c16) step (%c8)
// CHECK:           linalg.fill
// CHECK:           tensor.parallel_insert_slice

// -----

func.func @remove_empty_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = scf.forall (%arg4, %arg5) = (%c0, %c16) to (%c1, %c16)
        step (%c8, %c8) shared_outs (%out_ = %0) -> (tensor<8x8xf32>) {
    %11 = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %out_[%arg4, %arg5] [8, 8] [1, 1]
        : tensor<8x8xf32> into tensor<8x8xf32>
    }
  }
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @remove_empty_parallel
// CHECK-NOT:   scf.forall
// CHECK:       %[[EMPTY:.*]] = tensor.empty
// CHECK:       return %[[EMPTY]]
