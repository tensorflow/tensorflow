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
// CHECK:         %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[ARG0:.*]] = %{{.*}}, %[[ARG1:.*]] =
// CHECK:           %[[LHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<8x16xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<16x4xf32>, vector<2x4xf32>
// CHECK:           %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:        %[[LHS]], %[[RHS]], %[[ARG1]]
// CHECK:           scf.yield %[[ARG0]], %[[CONTRACT]]
// CHECK:         vector.transfer_write %[[FOR]]#1,  %[[FOR]]#0

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
// CHECK:         %[[FOR:.*]]:2 = scf.for {{.*}} iter_args(%[[ARG0:.*]] = %{{.*}}, %[[ARG1:.*]] = %[[OUT_READ]]
// CHECK-NOT:       linalg.matmul
// CHECK:           %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<128x16xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<16x64xf32>, vector<2x4xf32>
// CHECK-NOT:       vector.transfer_read
// CHECK:           %[[CONTRACT:.*]] = vector.contract {{{.*}}} %[[LHS]], %[[RHS]], %[[ARG1]]
// CHECK:           scf.yield %[[ARG0]], %[[CONTRACT]]
// CHECK:         vector.transfer_write  %[[FOR]]#1,  %[[FOR]]#0

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

func.func @dont_vectorize_any_ite(%arg0: i1, %arg1: tensor<8x1xf32>,
    %arg2: tensor<8x1xf32>) -> tensor<8x1xf32> {
  %0 = scf.if %arg0 -> (tensor<8x1xf32>) {
    scf.yield %arg1 : tensor<8x1xf32>
  } else {
    scf.yield %arg2 : tensor<8x1xf32>
  }
  return %0 : tensor<8x1xf32>
}

// CHECK-LABEL: @dont_vectorize_any_ite
// CHECK:         scf.if %{{.*}} -> (tensor<8x1xf32>)

// -----

func.func @vectorize_ite_w_vector_producers(%arg0: i1, %arg1: vector<8x1xf32>,
    %arg2: vector<8x1xf32>) -> tensor<8x1xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<8x1xf32>
  %1 = scf.if %arg0 -> (tensor<8x1xf32>) {
    %2 = vector.transfer_write %arg1, %0[%c0, %c0] {in_bounds = [true, true]}
        : vector<8x1xf32>, tensor<8x1xf32>
    scf.yield %2 : tensor<8x1xf32>
  } else {
    %2 = vector.transfer_write %arg2, %0[%c0, %c0] {in_bounds = [true, true]}
        : vector<8x1xf32>, tensor<8x1xf32>
    scf.yield %2 : tensor<8x1xf32>
  }
  return %1 : tensor<8x1xf32>
}

// CHECK-LABEL: @vectorize_ite_w_vector_producers
// CHECK-SAME:      %[[ARG0:.*]]: i1, %[[ARG1:.*]]: vector<8x1xf32>, %[[ARG2:.*]]: vector<8x1xf32>
// CHECK:         %[[IF:.*]] = scf.if %[[ARG0]] -> (vector<8x1xf32>)
// CHECK:           scf.yield %[[ARG1]]
// CHECK:         else
// CHECK:           scf.yield %[[ARG2]]
// CHECK:         %[[TRANSFER:.*]] = vector.transfer_write %[[IF]]
// CHECK:         return %[[TRANSFER]]

// -----

func.func @vectorize_ite_w_vector_users(%arg0: i1, %arg1: tensor<8x1xf32>,
    %arg2: tensor<8x1xf32>) -> vector<8x1xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.if %arg0 -> (tensor<8x1xf32>) {
    scf.yield %arg1 : tensor<8x1xf32>
  } else {
    scf.yield %arg2 : tensor<8x1xf32>
  }
  %1 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]}
      : tensor<8x1xf32>, vector<8x1xf32>
  return %1 : vector<8x1xf32>
}

// CHECK-LABEL: @vectorize_ite_w_vector_users
// CHECK-SAME:      %[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<8x1xf32>, %[[ARG2:.*]]: tensor<8x1xf32>
// CHECK:         %[[IF:.*]] = scf.if %[[ARG0]] -> (vector<8x1xf32>)
// CHECK:           %[[TRANSFER:.*]] = vector.transfer_read
// CHECK:           scf.yield %[[TRANSFER]] : vector<8x1xf32>
// CHECK:         else
// CHECK:           %[[TRANSFER_0:.*]] = vector.transfer_read
// CHECK:           scf.yield %[[TRANSFER_0]] : vector<8x1xf32>
// CHECK:         return %[[IF]]

// -----

func.func @dont_vectorize_complex_ite(%arg0: i1,
    %arg1: tensor<8x1xcomplex<f32>>, %arg2: tensor<8x1xcomplex<f32>>)
    -> tensor<8x1xcomplex<f32>> {
  %0 = scf.if %arg0 -> (tensor<8x1xcomplex<f32>>) {
    scf.yield %arg1 : tensor<8x1xcomplex<f32>>
  } else {
    scf.yield %arg2 : tensor<8x1xcomplex<f32>>
  }
  return %0 : tensor<8x1xcomplex<f32>>
}

// CHECK-LABEL: @dont_vectorize_complex_ite
// CHECK:         scf.if %{{.*}} -> (tensor<8x1xcomplex<f32>>)

// -----

func.func @vectorize_ite_w_scalar(%arg0: i1, %arg1: tensor<8x1xf32>, %arg2: f32,
    %arg3: tensor<8x1xf32>, %arg4: f32) -> (vector<8x1xf32>, f32) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0:2 = scf.if %arg0 -> (tensor<8x1xf32>, f32) {
    scf.yield %arg1, %arg2 : tensor<8x1xf32>, f32
  } else {
    scf.yield %arg3, %arg4 : tensor<8x1xf32>, f32
  }
  %1 = vector.transfer_read %0#0[%c0, %c0], %cst {in_bounds = [true, true]}
      : tensor<8x1xf32>, vector<8x1xf32>
  return %1, %0#1 : vector<8x1xf32>, f32
}

// CHECK-LABEL: @vectorize_ite_w_scalar
// CHECK-SAME:      %[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<8x1xf32>, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: tensor<8x1xf32>, %[[ARG4:.*]]: f32
// CHECK:         %[[IF:.*]]:2 = scf.if %[[ARG0]] -> (vector<8x1xf32>, f32)
// CHECK:           %[[TRANSFER:.*]] = vector.transfer_read %[[ARG1]]
// CHECK:           scf.yield %[[TRANSFER]], %[[ARG2]]
// CHECK:         else
// CHECK:           %[[TRANSFER_0:.*]] = vector.transfer_read %[[ARG3]]
// CHECK:           scf.yield %[[TRANSFER_0]], %[[ARG4]]
// CHECK:         return %[[IF]]#0, %[[IF]]#1

// -----

func.func @vectorize_ite_w_casts(%arg0: i1, %arg1: tensor<8x1xf32>,
    %arg2: tensor<8x1xf32>) -> vector<8x1xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.if %arg0 -> (tensor<?x1xf32>) {
    %cast_0 = tensor.cast %arg1 : tensor<8x1xf32> to tensor<?x1xf32>
    scf.yield %cast_0 : tensor<?x1xf32>
  } else {
    %cast_0 = tensor.cast %arg2 : tensor<8x1xf32> to tensor<?x1xf32>
    scf.yield %cast_0 : tensor<?x1xf32>
  }
  %cast = tensor.cast %0 : tensor<?x1xf32> to tensor<8x1xf32>
  %1 = vector.transfer_read %cast[%c0, %c0], %cst {in_bounds = [true, true]}
      : tensor<8x1xf32>, vector<8x1xf32>
  return %1 : vector<8x1xf32>
}

// -----

// CHECK-LABEL: @vectorize_ite_w_casts
// CHECK-SAME:      %[[ARG0:.*]]: i1, %[[ARG1:.*]]: tensor<8x1xf32>, %[[ARG2:.*]]: tensor<8x1xf32>
// CHECK:         %[[IF:.*]] = scf.if %[[ARG0]] -> (vector<8x1xf32>)
// CHECK:           %[[TRANSFER:.*]] = vector.transfer_read %[[ARG1]]
// CHECK:           scf.yield %[[TRANSFER]]
// CHECK:         else
// CHECK:           %[[TRANSFER_0:.*]] = vector.transfer_read %[[ARG2]]
// CHECK:           scf.yield %[[TRANSFER_0]]
// CHECK:         return %[[IF]]
