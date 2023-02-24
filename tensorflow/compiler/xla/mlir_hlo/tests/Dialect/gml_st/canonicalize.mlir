// RUN: mlir-hlo-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

func.func @inline_single_iteration_parallel(
    %in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c0) to (%c1, %c1)
        step (%c8, %c8) outs (%out_ = %0: tensor<8x8xf32>) {
    %20 = tensor.extract_slice %out_[%arg4, %arg5] [8, 8] [1, 1]
      : tensor<8x8xf32> to tensor<8x8xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%20 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    gml_st.set_yield %11 into %out_[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @inline_single_iteration_parallel
// CHECK-NOT:     gml_st.parallel
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
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c0) to (%c1, %c16)
        step (%c8, %c8) outs (%out_ = %0: tensor<8x8xf32>) {
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %out_[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @collapse_one_dim_parallel
// CHECK:         gml_st.parallel (%[[ARG:.*]]) = (%c0) to (%c16) step (%c8)
// CHECK:           gml_st.tile [0, %[[ARG]]]
// CHECK:           linalg.fill
// CHECK:           gml_st.set_yield

// -----

func.func @remove_empty_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c16) to (%c1, %c16)
        step (%c8, %c8) outs (%out_ = %0: tensor<8x8xf32>) {
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%out_ : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %out_[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @remove_empty_parallel
// CHECK-NOT:   gml_st.parallel
// CHECK:       %[[EMPTY:.*]] = tensor.empty
// CHECK:       return %[[EMPTY]]

// -----

func.func @fold_tensor_cast_into_parallel(
    %in: tensor<2xi32>, %out: tensor<2xi32>) -> tensor<2xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 100500 : i32

  %out_cast = tensor.cast %out : tensor<2xi32> to tensor<?xi32>
  %result = gml_st.parallel (%i) = (%c0) to (%c2) step (%c1)
      outs (%out_ = %out_cast: tensor<?xi32>) {
    %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
    gml_st.set_yield %cst into %out_[%tile]
      : i32 into tensor<?xi32>[!gml_st.tile<1>]
  } : tensor<?xi32>
  %result_cast = tensor.cast %result
    : tensor<?xi32> to tensor<2xi32>

  func.return %result_cast : tensor<2xi32>
}
// CHECK-LABEL: @fold_tensor_cast_into_parallel
// CHECK:         gml_st.parallel
// CHECK-NEXT:      gml_st.tile
// CHECK-NEXT:      gml_st.set_yield
// CHECK-SAME:        i32 into tensor<2xi32>
// CHECK-NEXT:    } : tensor<2xi32>
// CHECK-NEXT:    return

// -----

func.func @dim_of_parallel_loop(
    %in: tensor<2x10xi32>, %out: tensor<2x10xi32>) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 100500 : i32

  %result = gml_st.parallel (%i, %j) = (%c0, %c0) to (%c2, %c10)
      step (%c1, %c1) outs (%out_ = %out: tensor<2x10xi32>) {
    %tile = gml_st.tile [%i, %j] [1, 1] [1, 1] : !gml_st.tile<1x1>
    gml_st.set_yield %cst into %out_[%tile]
      : i32 into tensor<2x10xi32>[!gml_st.tile<1x1>]
  } : tensor<2x10xi32>

  %dim = tensor.dim %result, %c1 : tensor<2x10xi32>
  func.return %dim : index
}
// CHECK-LABEL: @dim_of_parallel_loop
// CHECK:         %[[C10:.*]] = arith.constant 10
// CHECK-NEXT:    return %[[C10]]
