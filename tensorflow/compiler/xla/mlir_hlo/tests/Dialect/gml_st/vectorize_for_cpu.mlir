// RUN: mlir-hlo-opt %s --vectorize-for-cpu --split-input-file |\
// RUN: FileCheck %s


func.func @vectorize_tiled_matmul(%lhs: tensor<8x16xf32>,
    %rhs: tensor<16x4xf32>, %fill: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index

  %7 = scf.for %i = %c0 to %c16 step %c2
      iter_args (%arg6 = %fill) -> (tensor<8x4xf32>) {
    %9 = tensor.extract_slice %lhs[0, %i] [8, 2] [1, 1]  :
              tensor<8x16xf32> to tensor<8x2xf32>

    %11 = tensor.extract_slice %rhs[%i, 0] [2, 4] [1, 1]  :
              tensor<16x4xf32> to tensor<2x4xf32>

    %13 = tensor.extract_slice %arg6[0, 0] [8, 4] [1, 1]  :
              tensor<8x4xf32> to tensor<8x4xf32>

    %14 = linalg.matmul ins(%9, %11 : tensor<8x2xf32>, tensor<2x4xf32>)
                        outs(%13 : tensor<8x4xf32>) -> tensor<8x4xf32>

    %12 = tensor.insert_slice %14 into %arg6 [0, 0] [8, 4] [1, 1]
      : tensor<8x4xf32> into tensor<8x4xf32>

    scf.yield %14 : tensor<8x4xf32>
  } {__perfectly_tileable_loop_label__}
  return %7 : tensor<8x4xf32>
}

// CHECK-LABEL: func @vectorize_tiled_matmul

// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[OUT:.*]]
// CHECK:         %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] =
// CHECK:           %[[LHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<8x16xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<16x4xf32>, vector<2x4xf32>
// CHECK:           %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:        %[[LHS]], %[[RHS]], %[[ARG]]
// CHECK:           scf.yield %[[CONTRACT]]
// CHECK:         vector.transfer_write %[[FOR]]

// -----

func.func @vectorize_static_matmul(%lhs: tensor<128x16xf32>,
    %rhs: tensor<16x64xf32>, %fill: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %0 = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c128, %c64) step (%c8, %c4)
    outs (%out_ = %fill: tensor<128x64xf32>) {
    %2 = tensor.extract_slice %lhs[%i, 0] [8, 16] [1, 1] :
            tensor<128x16xf32> to tensor<8x16xf32>
    %4 = tensor.extract_slice %rhs[0, %j] [16, 4] [1, 1] :
            tensor<16x64xf32> to tensor<16x4xf32>
    %6 = tensor.extract_slice %fill[%i, %j] [8, 4] [1, 1] :
            tensor<128x64xf32> to tensor<8x4xf32>
    %7 = scf.for %k = %c0 to %c16 step %c2 iter_args (%arg6 = %6) -> (tensor<8x4xf32>) {
      %9 = tensor.extract_slice %2[0, %k] [8, 2] [1, 1] :
                tensor<8x16xf32> to tensor<8x2xf32>
      %11 = tensor.extract_slice %4[%k, 0] [2, 4] [1, 1] :
                tensor<16x4xf32> to tensor<2x4xf32>
      %13 = tensor.extract_slice %arg6[0, 0] [8, 4] [1, 1] :
                tensor<8x4xf32> to tensor<8x4xf32>
      %14 = linalg.matmul ins(%9, %11 : tensor<8x2xf32>, tensor<2x4xf32>)
                          outs(%13 : tensor<8x4xf32>) -> tensor<8x4xf32>
      scf.yield %14 : tensor<8x4xf32>
    }
    %5 = gml_st.tile [%i, %j] [8, 4] [1, 1] : !gml_st.tile<8x4>
    gml_st.set_yield %7 into %out_[%5] :
            tensor<8x4xf32> into tensor<128x64xf32>[!gml_st.tile<8x4>]
  } : tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
// CHECK-LABEL: func @vectorize_static_matmul

// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read {{.*}} : tensor<8x4xf32>, vector<8x4xf32>
// CHECK:         %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[OUT_READ]]
// CHECK-NOT:       linalg.matmul
// CHECK:           %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<128x16xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<16x64xf32>, vector<2x4xf32>
// CHECK-NOT:       vector.transfer_read
// CHECK:           %[[CONTRACT:.*]] = vector.contract {{{.*}}} %[[LHS]], %[[RHS]], %[[ARG]]
// CHECK:           scf.yield %[[CONTRACT]]
// CHECK:         vector.transfer_write %[[FOR]]

// -----

func.func @transpose(%input: tensor<4x5x6xf32>,
    %init: tensor<5x6x4xf32>) -> tensor<5x6x4xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<4x5x6xf32>)
    outs(%init:tensor<5x6x4xf32>)
    permutation = [1, 2, 0]
  func.return %transpose : tensor<5x6x4xf32>
}

// CHECK-LABEL: func @transpose(
// CHECK-SAME:  %[[INPUT:.*]]: tensor<4x5x6xf32>
// CHECK-SAME:  %[[INIT:.*]]: tensor<5x6x4xf32>

// CHECK:         %[[READ:.*]] = vector.transfer_read %[[INPUT]]
// CHECK:         %[[TRANSPOSE:.*]] = vector.transpose %[[READ]], [1, 2, 0]
// CHECK:         %[[WRITE:.*]] = vector.transfer_write %[[TRANSPOSE]], %[[INIT]]
// CHECK:         return %[[WRITE]]

// -----

func.func @simplify_identity_transpose(%input: tensor<1x1xf32>,
    %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<1x1xf32>)
    outs(%init:tensor<1x1xf32>)
    permutation = [0, 1]
  func.return %transpose : tensor<1x1xf32>
}

// CHECK-LABEL: func @simplify_identity_transpose(

// CHECK-NOT:     linalg.transpose
// CHECK:         return

// -----

func.func @do_not_simplify_transpose(%input: tensor<1x1xf32>,
    %init: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<1x1xf32>)
    outs(%init:tensor<1x1xf32>)
    permutation = [1, 0]
  func.return %transpose : tensor<1x1xf32>
}

// CHECK-LABEL: func @do_not_simplify_transpose(

// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK:         return %[[TRANSPOSE]]

// -----

func.func @perfectly_tiled_reverse_1d(%input: tensor<8xf32>,
    %init: tensor<8xf32>) -> tensor<8xf32> {
  %res = thlo.reverse
         ins(%input: tensor<8xf32>)
         outs(%init: tensor<8xf32>)
         reverse_dimensions = [0]
  func.return %res : tensor<8xf32>
}

// CHECK-LABEL: func @perfectly_tiled_reverse_1d(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<8xf32>, %[[ARG1:.*]]: tensor<8xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]]
//       CHECK:   %[[SHUFFLE:.*]] = vector.shuffle %[[READ]]
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[SHUFFLE]], %[[ARG1]]
//       CHECK:   return %[[WRITE]]

// -----

func.func @perfectly_tiled_reverse_2d(%input: tensor<1x8xf32>,
    %init: tensor<1x8xf32>) -> tensor<1x8xf32> {
  %res = thlo.reverse
         ins(%input: tensor<1x8xf32>)
         outs(%init: tensor<1x8xf32>)
         reverse_dimensions = [1]
  func.return %res : tensor<1x8xf32>
}

// CHECK-LABEL: func @perfectly_tiled_reverse_2d(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x8xf32>, %[[ARG1:.*]]: tensor<1x8xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME:   : tensor<1x8xf32>, vector<8xf32>
//       CHECK:   %[[SHUFFLE:.*]] = vector.shuffle %[[READ]]
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[SHUFFLE]], %[[ARG1]]
//  CHECK-SAME:   : vector<8xf32>, tensor<1x8xf32>
//       CHECK:   return %[[WRITE]]

// -----

func.func @perfectly_tiled_reverse_4d(%input: tensor<1x1x1x8xf32>,
    %init: tensor<1x1x1x8xf32>) -> tensor<1x1x1x8xf32> {
  %res = thlo.reverse
         ins(%input: tensor<1x1x1x8xf32>)
         outs(%init: tensor<1x1x1x8xf32>)
         reverse_dimensions = [3]
  func.return %res : tensor<1x1x1x8xf32>
}

// CHECK-LABEL: func @perfectly_tiled_reverse_4d(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x1x1x8xf32>, %[[ARG1:.*]]: tensor<1x1x1x8xf32>
//       CHECK:   %[[READ:.*]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME:   : tensor<1x1x1x8xf32>, vector<8xf32>
//       CHECK:   %[[SHUFFLE:.*]] = vector.shuffle %[[READ]]
//       CHECK:   %[[WRITE:.*]] = vector.transfer_write %[[SHUFFLE]], %[[ARG1]]
//  CHECK-SAME:   : vector<8xf32>, tensor<1x1x1x8xf32>
//       CHECK:   return %[[WRITE]]

// -----

func.func @matvec(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @matvec
// CHECK-SAME:  %[[LHS:.*]]: tensor<33x17xf32>, %[[RHS:.*]]: tensor<17xf32>, %[[OUT:.*]]: tensor<33xf32>
// CHECK:         %[[LHS_READ:.*]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_READ:.*]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[CONTRACT:.*]] = vector.contract {{.*}}%[[LHS_READ]], %[[RHS_READ]], %[[OUT_READ]]
// CHECK:         vector.transfer_write %[[CONTRACT]], %[[OUT]]

// -----

func.func @vecmat(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat
// CHECK-SAME:  %[[LHS:.*]]: tensor<17xf32>, %[[RHS:.*]]: tensor<17x33xf32>, %[[OUT:.*]]: tensor<33xf32>
// CHECK:         %[[LHS_READ:.*]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_READ:.*]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[CONTRACT:.*]] = vector.contract {{.*}}%[[LHS_READ]], %[[RHS_READ]], %[[OUT_READ]]
// CHECK:         vector.transfer_write %[[CONTRACT]], %[[OUT]]

// -----

func.func @dot(%lhs: tensor<17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<f32>) -> tensor<f32> {
  %2 = linalg.dot ins(%lhs, %rhs : tensor<17xf32>, tensor<17xf32>)
                     outs(%output : tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: @dot
// CHECK-SAME:  %[[LHS:.*]]: tensor<17xf32>, %[[RHS:.*]]: tensor<17xf32>, %[[OUT:.*]]: tensor<f32>
// CHECK:         %[[LHS_READ:.*]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_READ:.*]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[OUT]]
// CHECK:         %[[CONTRACT:.*]] = vector.contract {{.*}}%[[LHS_READ]], %[[RHS_READ]]
// CHECK:         vector.transfer_write {{.*}}, %[[OUT]]

// -----

func.func @vectorize_ite(%pred: i1, %lhs: tensor<8x1xf32>,
    %rhs: tensor<8x1xf32>) -> tensor<8x1xf32> {
  %0 = scf.if %pred -> (tensor<8x1xf32>) {
    scf.yield %lhs : tensor<8x1xf32>
  } else {
    scf.yield %rhs : tensor<8x1xf32>
  }
  return %0 : tensor<8x1xf32>
}

// CHECK-LABEL:  @vectorize_ite
// REENABLE-SAME:       %[[PRED:.*]]: i1, %[[LHS:.*]]: tensor<8x1xf32>, %[[RHS:.*]]: tensor<8x1xf32>
// REENABLE-DAG:      %[[C0:.*]] = arith.constant 0 : index
// REENABLE-DAG:      %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// REENABLE:          %[[IF:.*]] = scf.if %[[PRED]] -> (vector<8x1xf32>)
// REENABLE:            %[[TRANSFER:.*]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]], %[[ZERO]]
// REENABLE:            scf.yield %[[TRANSFER]]
// REENABLE:          else
// REENABLE:            %[[TRANSFER_0:.*]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]]], %[[ZERO]]
// REENABLE:            scf.yield %[[TRANSFER_0]]
// REENABLE:          %[[EMPTY:.*]] = tensor.empty
// REENABLE:          %[[TRANSFER_1:.*]] = vector.transfer_write %[[IF]], %[[EMPTY]][%[[C0]], %[[C0]]]
// REENABLE:          return %[[TRANSFER_1]]

// -----

func.func @vectorize_ite_and_scalar(%pred: i1, %lhs: tensor<8x1xf32>,
    %lhs_scalar: f32, %rhs: tensor<8x1xf32>, %rhs_scalar: f32)
    -> (tensor<8x1xf32>, f32) {
  %0:2 = scf.if %pred -> (tensor<8x1xf32>, f32) {
    scf.yield %lhs, %lhs_scalar: tensor<8x1xf32>, f32
  } else {
    scf.yield %rhs, %rhs_scalar : tensor<8x1xf32>, f32
  }
  return %0#0, %0#1 : tensor<8x1xf32>, f32
}

// CHECK-LABEL:  @vectorize_ite_and_scalar
// REENABLE-SAME:       %[[PRED:.*]]: i1, %[[LHS:.*]]: tensor<8x1xf32>, %[[LHS_SCALAR:.*]]: f32, %[[RHS:.*]]: tensor<8x1xf32>, %[[RHS_SCALAR:.*]]: f32
// REENABLE-DAG:      %[[C0:.*]] = arith.constant 0 : index
// REENABLE-DAG:      %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// REENABLE:          %[[IF:.*]]:2 = scf.if %[[PRED]] -> (vector<8x1xf32>, f32)
// REENABLE:            %[[TRANSFER:.*]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]], %[[CST]]
// REENABLE:            scf.yield %[[TRANSFER]], %[[LHS_SCALAR]]
// REENABLE:          else
// REENABLE:            %[[TRANSFER_0:.*]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]]], %[[CST]]
// REENABLE:            scf.yield %[[TRANSFER_0]], %[[RHS_SCALAR]]
// REENABLE:          %[[EMPTY:.*]] = tensor.empty
// REENABLE:          %[[TRANSFER_1:.*]] = vector.transfer_write %[[IF]]#0, %[[EMPTY]][%[[C0]], %[[C0]]]
// REENABLE:          return %[[TRANSFER_1]], %[[IF]]#1

// -----

func.func @vectorize_ite_w_casts(%pred: i1, %lhs: tensor<8x1xf32>,
    %rhs: tensor<8x1xf32>) -> tensor<8x1xf32> {
  %0 = scf.if %pred -> (tensor<?x1xf32>) {
    %lhs_ = tensor.cast %lhs : tensor<8x1xf32> to tensor<?x1xf32>
    scf.yield %lhs_ : tensor<?x1xf32>
  } else {
    %rhs_ = tensor.cast %rhs : tensor<8x1xf32> to tensor<?x1xf32>
    scf.yield %rhs_ : tensor<?x1xf32>
  }
  %1 = tensor.cast %0 : tensor<?x1xf32> to tensor<8x1xf32>
  return %1 : tensor<8x1xf32>
}

// CHECK-LABEL:  @vectorize_ite_w_casts
// REENABLE-SAME:       %[[PRED:.*]]: i1, %[[LHS:.*]]: tensor<8x1xf32>, %[[RHS:.*]]: tensor<8x1xf32>
// REENABLE:          %[[RES:.*]] = scf.if %[[PRED]]
// REENABLE-SAME:         vector<8x1xf32>
// REENABLE:            %[[LHS_:.*]] = vector.transfer_read %[[LHS]]
// REENABLE:            scf.yield %[[LHS_]]
// REENABLE:          else
// REENABLE:            %[[RHS_:.*]] = vector.transfer_read %[[RHS]]
// REENABLE:            scf.yield %[[RHS_]]
// REENABLE:          %[[RES_:.*]] = vector.transfer_write %[[RES]]
// REENABLE:          return %[[RES_]]
