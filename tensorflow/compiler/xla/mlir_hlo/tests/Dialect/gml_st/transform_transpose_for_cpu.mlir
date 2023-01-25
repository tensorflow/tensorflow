// RUN: mlir-hlo-opt %s --gml-st-cpu-transform-transpose | FileCheck %s

func.func @transpose_permutation(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<16x32x64xf32>)
    outs(%init:tensor<32x64x16xf32>)
    permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}

// CHECK-LABEL: func.func @transpose_permutation(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<16x32x64xf32>,
// CHECK-SAME:      %[[INIT:.*]]: tensor<32x64x16xf32>)

// CHECK:             gml_st.parallel
// CHECK:               %[[INPUT_SUB:.*]] = tensor.extract_slice %[[INPUT]]
// CHECK:      : tensor<16x32x64xf32> to tensor<8x1x8xf32>

// CHECK:              %[[INIT_SUB:.*]] =  tensor.extract_slice %[[INIT]]
// CHECK:      : tensor<32x64x16xf32> to tensor<1x8x8xf32>

// CHECK:       linalg.transpose
// CHECK-SAME:    ins(%[[INPUT_SUB]] : tensor<8x1x8xf32>)
// CHECK-SAME:    outs(%[[INIT_SUB]] : tensor<1x8x8xf32>)
// CHECK-SAME:    permutation = [1, 2, 0]

// -----

func.func @peel_transpose(%input: tensor<16x32x65xf32>,
    %init: tensor<32x65x16xf32>) -> tensor<32x65x16xf32> {
  %transpose = linalg.transpose
    ins(%input:tensor<16x32x65xf32>)
    outs(%init:tensor<32x65x16xf32>)
    permutation = [1, 2, 0]
  func.return %transpose : tensor<32x65x16xf32>
}

// CHECK-LABEL: @peel_transpose(
// CHECK-SAME: %[[INPUT:.*]]: tensor<16x32x65xf32>
// CHECK-SAME: %[[INIT:.*]]: tensor<32x65x16xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 :
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 :
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 :
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 :
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 :
// CHECK-DAG: %[[C65:.*]] = arith.constant 65 :
// CHECK: gml_st.parallel {{.*}} (%[[C0]], %[[C0]], %[[C0]]) to (%[[C16]], %[[C32]], %[[C64]])
// CHECK:   linalg.transpose ins({{.*}} tensor<8x1x?xf32>) outs({{.*}} tensor<1x?x8xf32>)
// CHECK: gml_st.parallel {{.*}} (%[[C0]], %[[C0]], %[[C64]]) to (%[[C16]], %[[C32]], %[[C65]])
// CHECK:   gml_st.parallel {{.*}} (%[[C0]], %[[C0]], %[[C0]]) {{.*}} step (%[[C1]], %[[C1]], %[[C1]])
// CHECK:     linalg.transpose ins({{.*}} tensor<1x1x1xf32>) outs({{.*}} tensor<1x1x1xf32>)

// -----

func.func @do_not_tile_inside_scf_for(%input: tensor<16x16xf32>,
    %init: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %10 = scf.for %arg2 = %c0 to %c16 step %c8 iter_args(%arg3 = %init) -> (tensor<16x16xf32>) {
    %11 = scf.for %arg4 = %c0 to %c16 step %c8 iter_args(%arg5 = %arg3) -> (tensor<16x16xf32>) {
      %extracted_slice_1 = tensor.extract_slice %input[%arg2, %arg4] [8, 8] [1, 1] : tensor<16x16xf32> to tensor<8x8xf32>
      %14 = tensor.empty() : tensor<8x8xf32>
      %transposed = linalg.transpose ins(%extracted_slice_1 : tensor<8x8xf32>) outs(%14 : tensor<8x8xf32>) permutation = [0, 1]
      %inserted_slice = tensor.insert_slice %transposed into %arg5[%arg2, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<16x16xf32>
      scf.yield %inserted_slice : tensor<16x16xf32>
    }
    scf.yield %11 : tensor<16x16xf32>
  }
  return %10 : tensor<16x16xf32>
}

// CHECK-LABEL: @do_not_tile_inside_scf_for(

// CHECK:     scf.for
// CHECK:       scf.for
// CHECK-NOT: gml_st.parallel
// CHECK:       linalg.transpose
