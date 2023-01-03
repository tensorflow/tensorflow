// RUN: mlir-hlo-opt %s --vectorize-perfectly-tiled-loops --split-input-file |\
// RUN: FileCheck %s


func.func @vectorize_tiled_matmul(%lhs: tensor<8x16xf32>,
    %rhs: tensor<16x4xf32>, %fill: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index

  %7 = gml_st.for (%i) =
              (%c0) to (%c16) step (%c2) outs (%arg6 = %fill: tensor<8x4xf32>) {
    %9 = gml_st.materialize %lhs[0, %i] [8, 2] [1, 1]  :
              tensor<8x16xf32> to tensor<8x2xf32>

    %11 = gml_st.materialize %rhs[%i, 0] [2, 4] [1, 1]  :
              tensor<16x4xf32> to tensor<2x4xf32>

    %13 = gml_st.materialize %arg6[0, 0] [8, 4] [1, 1]  :
              tensor<8x4xf32> to tensor<8x4xf32>

    %14 = linalg.matmul ins(%9, %11 : tensor<8x2xf32>, tensor<2x4xf32>)
                        outs(%13 : tensor<8x4xf32>) -> tensor<8x4xf32>

    %12 = gml_st.tile [0, 0] [8, 4] [1, 1] : !gml_st.tile<8x4>
    gml_st.set_yield %14 into %arg6[%12] :
              tensor<8x4xf32> into tensor<8x4xf32>[!gml_st.tile<8x4>]
  } {__perfectly_tileable_loop_label__} : tensor<8x4xf32>
  return %7 : tensor<8x4xf32>
}

// CHECK-LABEL: func @vectorize_tiled_matmul

// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[OUT:.*]]
// CHECK:         %[[FOR:.*]] = gml_st.for {{.*}} outs (%[[ARG:.*]] =
// CHECK:           %[[LHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<8x2xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read
// CHECK-SAME:        : tensor<2x4xf32>, vector<2x4xf32>
// CHECK:           %[[CONTRACT:.*]] = vector.contract
// CHECK-SAME:        %[[LHS]], %[[RHS]], %[[ARG]]
// CHECK:           gml_st.set_yield %[[CONTRACT]] into %[[ARG]]
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
  %0 = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c128, %c64) step (%c8, %c4) {
    %2 = gml_st.materialize %lhs[%i, 0] [8, 16] [1, 1] :
            tensor<128x16xf32> to tensor<8x16xf32>
    %4 = gml_st.materialize %rhs[0, %j] [16, 4] [1, 1] :
            tensor<16x64xf32> to tensor<16x4xf32>
    %6 = gml_st.materialize %fill[%i, %j] [8, 4] [1, 1] :
            tensor<128x64xf32> to tensor<8x4xf32>
    %7 = gml_st.for (%k) =
                (%c0) to (%c16) step (%c2) outs (%arg6 = %6: tensor<8x4xf32>) {
      %9 = gml_st.materialize %2[0, %k] [8, 2] [1, 1] :
                tensor<8x16xf32> to tensor<8x2xf32>
      %11 = gml_st.materialize %4[%k, 0] [2, 4] [1, 1] :
                tensor<16x4xf32> to tensor<2x4xf32>
      %13 = gml_st.materialize %arg6[0, 0] [8, 4] [1, 1] :
                tensor<8x4xf32> to tensor<8x4xf32>
      %14 = linalg.matmul ins(%9, %11 : tensor<8x2xf32>, tensor<2x4xf32>)
                          outs(%13 : tensor<8x4xf32>) -> tensor<8x4xf32>
      %12 = gml_st.tile [0, 0] [8, 4] [1, 1] : !gml_st.tile<8x4>
      gml_st.set_yield %14 into %arg6[%12] :
                tensor<8x4xf32> into tensor<8x4xf32>[!gml_st.tile<8x4>]
    } : tensor<8x4xf32>
    %5 = gml_st.tile [%i, %j] [8, 4] [1, 1] : !gml_st.tile<8x4>
    gml_st.set_yield %7 into %fill[%5] :
            tensor<8x4xf32> into tensor<128x64xf32>[!gml_st.tile<8x4>]
  } : tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
// CHECK-LABEL: func @vectorize_static_matmul

// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read {{.*}} : tensor<8x4xf32>, vector<8x4xf32>
// CHECK:         %[[FOR:.*]] = gml_st.for {{.*}} outs (%[[ARG:.*]] = %[[OUT_READ]]
// CHECK-NOT:       linalg.matmul
// CHECK:           %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<8x2xf32>, vector<8x2xf32>
// CHECK:           %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<2x4xf32>, vector<2x4xf32>
// CHECK-NOT:       vector.transfer_read
// CHECK:           %[[CONTRACT:.*]] = vector.contract {{{.*}}} %[[LHS]], %[[RHS]], %[[ARG]]
// CHECK:           gml_st.set_yield %[[CONTRACT]] into %[[ARG]]
// CHECK:         vector.transfer_write %[[FOR]]

// -----

func.func @do_not_vectorize_materialize_outside_loop() -> tensor<8x1xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<10x1xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<10x1xf32>) -> tensor<10x1xf32>
  %3 = gml_st.materialize %1[0, 0] [8, 1] [1, 1]  : tensor<10x1xf32> to tensor<8x1xf32>
  %4 = gml_st.loop (%arg2, %arg3) = (%c0, %c0) to (%c8, %c1) step (%c1, %c8) ins (%arg4 = %cst: f32) outs (%arg5 = %3: tensor<8x1xf32>) {
    %10 = affine.min affine_map<(d0) -> (-d0 + 1, 8)>(%arg3)
    %extracted_slice = tensor.extract_slice %arg5[%arg2, %arg3] [1, %10] [1, 1] : tensor<8x1xf32> to tensor<1x?xf32>
    %11 = linalg.fill ins(%arg4 : f32) outs(%extracted_slice : tensor<1x?xf32>) -> tensor<1x?xf32>
    %inserted_slice_1 = tensor.insert_slice %11 into %arg5[%arg2, %arg3] [1, %10] [1, 1] : tensor<1x?xf32> into tensor<8x1xf32>
    gml_st.yield %inserted_slice_1 : tensor<8x1xf32>
  }
  return %4 : tensor<8x1xf32>
}
// CHECK-LABEL: func @do_not_vectorize_materialize_outside_loop
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<10x1xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty() : tensor<10x1xf32>
// CHECK:         %[[WRITE:.*]] = vector.transfer_write %[[CST]], %[[INIT]]{{.*}} tensor<10x1xf32>
// CHECK:         gml_st.materialize %[[WRITE]] [0, 0] [8, 1] [1, 1] : {{.*}} to tensor<8x1xf32>

// -----

func.func @pad(%arg0: tensor<10x10xf32>) -> tensor<16x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0, 0] high[6, 0] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<10x10xf32> to tensor<16x10xf32>

  return %padded : tensor<16x10xf32>
}

// CHECK-LABEL: func @pad(

// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<16x10xf32>
// CHECK:         %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[EMPTY]]
// CHECK:         %[[READ:.*]] = vector.transfer_read
// CHECK:         %[[WRITE:.*]] = vector.transfer_write %[[READ]], %[[FILL]]
// CHECK:         return %[[WRITE]]

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
