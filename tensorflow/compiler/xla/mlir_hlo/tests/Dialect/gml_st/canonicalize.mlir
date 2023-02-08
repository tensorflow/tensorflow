// RUN: mlir-hlo-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// CHECK-LABEL: @fold_unit_dim
func.func @fold_unit_dim() -> tensor<8x10xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  %init = tensor.empty() : tensor<8x10xf32>
  // CHECK: gml_st.for (%[[I:.*]]) = (%[[C0]]) to (%[[C4]]) step (%[[C1]])
  %out = gml_st.for (%i, %j) = (%c0, %c4) to (%c4, %c5) step (%c1, %c1)
      outs(%out_ = %init : tensor<8x10xf32>) {
    // CHECK: gml_st.tile [%[[I]], 4]
     %tile = gml_st.tile [%i, %j] [4, 1] [1, 1] : !gml_st.tile<4x1>

     %val = tensor.empty() : tensor<4x1xf32>
     gml_st.set_yield %val into %out_[%tile]
      : tensor<4x1xf32> into tensor<8x10xf32>[!gml_st.tile<4x1>]
  } : tensor<8x10xf32>
  func.return %out : tensor<8x10xf32>
}

// -----

// CHECK-LABEL: @remove_empty_for
func.func @remove_empty_for() -> tensor<8x10xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  // CHECK: %[[INIT:.*]] = tensor.empty
  %init = tensor.empty() : tensor<8x10xf32>
  // CHECK-NOT: gml_st.for
  %out = gml_st.for (%i, %j) = (%c0, %c4) to (%c4, %c4) step (%c1, %c1)
      outs(%out_ = %init : tensor<8x10xf32>) {
     %tile = gml_st.tile [%i, %j] [4, 1] [1, 1] : !gml_st.tile<4x1>

     %val = tensor.empty() : tensor<4x1xf32>
     gml_st.set_yield %val into %out_[%tile]
      : tensor<4x1xf32> into tensor<8x10xf32>[!gml_st.tile<4x1>]
  } : tensor<8x10xf32>
  // CHECK: return %[[INIT]]
  func.return %out : tensor<8x10xf32>
}

// -----

// CHECK-LABEL: @fold_constant_tile_through_materialize
func.func @fold_constant_tile_through_materialize(%in: tensor<4xf32>) ->
    tensor<?xf32> {
  %c2 = arith.constant 2 : index
  // CHECK:       %[[MAT:.*]] = tensor.extract_slice
  // CHECK-SAME:    [2] [2] [1] : tensor<4xf32> to tensor<2xf32>
  %mat = tensor.extract_slice %in[%c2] [%c2] [1] : tensor<4xf32> to tensor<?xf32>
  // CHECK: %[[RET:.*]] = tensor.cast %[[MAT]] : tensor<2xf32> to tensor<?xf32>
  // CHECK: return %[[RET]]
  func.return %mat : tensor<?xf32>
}

// -----

func.func @fold_constant_set_yield(%in: tensor<?x?xf32>,
                                   %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = gml_st.for (%arg0) = (%c0) to (%c8) step (%c2)
                  outs (%arg1 = %out: tensor<?x?xf32>) {
    %out_sub = tensor.extract_slice %out[0, 0] [%c2, %c2] [1, 1]  :
                    tensor<?x?xf32> to tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32)
                        outs(%out_sub : tensor<?x?xf32>) -> tensor<?x?xf32>
    %tile = gml_st.tile [0, 0] [%c2, %c2] [1, 1] : !gml_st.tile<?x?>
    gml_st.set_yield %fill into %arg1[%tile] :
                    tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @fold_constant_set_yield
// CHECK:         %[[FOR:.*]] = gml_st.for{{.*}}: tensor<?x?xf32>
// CHECK-NEXT:      %[[SLICE:.*]] = tensor.extract_slice
// CHECK-SAME:        [0, 0] [2, 2]
// CHECK-NEXT:      %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[SLICE]] : tensor<2x2xf32>)
// CHECK-NEXT:      %[[TILE:.*]] = gml_st.tile [0, 0] [2, 2] {{.*}} !gml_st.tile<2x2>
// CHECK-NEXT:      gml_st.set_yield %[[FILL]] into %{{.*}}[%[[TILE]]] : tensor<2x2xf32> into tensor<?x?xf32>[!gml_st.tile<2x2>]

// -----

func.func @fold_constant_set_yield_scalar(%in: tensor<?xf32>,
                                   %out: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = gml_st.for (%arg0) = (%c0) to (%c8) step (%c2)
                  outs (%arg1 = %out: tensor<?xf32>) {
    %tile_1d = gml_st.tile [0] [%c1] [1] : !gml_st.tile<?>
    %cast = builtin.unrealized_conversion_cast %tile_1d :
                    !gml_st.tile<?> to !gml_st.tile<1>
    gml_st.set_yield %cst into %arg1[%cast] :
                    f32 into tensor<?xf32>[!gml_st.tile<1>]
  } : tensor<?xf32>
  return %1 : tensor<?xf32>
}

// CHECK-LABEL: @fold_constant_set_yield_scalar
// CHECK:         %[[FOR:.*]] = gml_st.for (%{{.*}}) outs
// CHECK-SAME:      (%[[INIT_:.*]] = %[[INIT:.*]]: tensor<?xf32>)
// CHECK:           %[[TILE:.*]] = gml_st.tile [0] [1] [1] : !gml_st.tile<1>
// CHECK-NOT:       builtin.unrealized_conversion_cast
// CHECK:           gml_st.set_yield %[[SCALAR:.*]] into %[[INIT_]][%[[TILE]]] : f32 into tensor<?xf32>[!gml_st.tile<1>]

// -----

func.func @fold_constant_for(%in: tensor<?x?xf32>,
                             %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %3 = tensor.extract_slice %out[0, 0] [8, 2] [1, 1] :
                  tensor<?x?xf32> to tensor<8x2xf32>
  %cast_3 = tensor.cast %3 : tensor<8x2xf32> to tensor<?x?xf32>
  %4 = gml_st.for (%arg0) = (%c0) to (%c8) step (%c2)
                  outs (%arg1 = %cast_3: tensor<?x?xf32>) {
    %out_sub = tensor.extract_slice %arg1[0, %arg0] [8, 2] [1, 1]  :
                    tensor<?x?xf32> to tensor<8x2xf32>
    %fill = linalg.fill ins(%cst : f32)
                        outs(%out_sub : tensor<8x2xf32>) -> tensor<8x2xf32>
    %cast_fill = tensor.cast %fill : tensor<8x2xf32> to tensor<?x?xf32>
    %tile = gml_st.tile [0, %arg0] [8, 2] [1, 1] : !gml_st.tile<8x2>
    %2 = builtin.unrealized_conversion_cast %tile :
                    !gml_st.tile<8x2> to !gml_st.tile<?x?>
    gml_st.set_yield %cast_fill into %arg1[%2] :
                    tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// CHECK-LABEL: @fold_constant_for
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice {{.*}} to tensor<8x2xf32>
// CHECK-NOT:     tensor.cast
// CHECK:         %[[FOR1:.*]] = gml_st.for (%[[I:.*]]) = (%c0) to {{.*}} outs (%[[ARG1:.*]] = %[[SLICE]]: tensor<8x2xf32>
// CHECK-NEXT:      %[[FOR1_SLICE:.*]] = tensor.extract_slice %[[ARG1]][0, %[[I]]] [8, 2] [1, 1] : tensor<8x2xf32> to tensor<8x2xf32>
// CHECK:           %[[FOR1_TILE:.*]] = gml_st.tile {{.*}} [8, 2] {{.*}} !gml_st.tile<8x2>
// CHECK:           gml_st.set_yield %{{.*}} into %[[ARG1]][%[[FOR1_TILE]]] : tensor<8x2xf32> into tensor<8x2xf32>[!gml_st.tile<8x2>]
// CHECK-NEXT:    } : tensor<8x2xf32>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[FOR1]] : tensor<8x2xf32> to tensor<?x?xf32>
// CHECK-NEXT:    return %[[CAST]] : tensor<?x?xf32>

// -----

func.func @fold_cast_to_materialize_source(%in: tensor<4xf32>) ->
    tensor<2xf32> {
  %cast = tensor.cast %in : tensor<4xf32> to tensor<?xf32>
  %mat = tensor.extract_slice %cast[2] [2] [1]
    : tensor<?xf32> to tensor<2xf32>
  func.return %mat : tensor<2xf32>
}

// CHECK-LABEL: @fold_cast_to_materialize_source
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>
// CHECK-NOT:     tensor.cast
// CHECK:         %[[MAT:.*]] = tensor.extract_slice %[[IN]][2] [2] [1] : tensor<4xf32> to tensor<2xf32>
// CHECK:         return %[[MAT]]

// -----

func.func @inline_single_iteration_for(
    %in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.for (%arg4) = (%c0) to (%c1) step (%c8)
        outs (%arg5 = %0: tensor<8x8xf32>) {
    %19 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%arg5 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %arg5[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @inline_single_iteration_for
// CHECK-NOT:     gml_st.for
// CHECK:         linalg.fill

// -----

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

func.func @fold_for_iter_arg(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = tensor.empty() : tensor<8x8xf32>
  %13:2 = gml_st.for (%arg4) = (%c0) to (%c16) step (%c8)
        outs (%arg5 = %0: tensor<8x8xf32>, %arg6 = %1: tensor<8x8xf32>) {
    %19 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%arg5 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %arg5[%19]
      : tensor<8x8xf32> into tensor<8x8xf32>[!gml_st.tile<8x8>],
                     %arg6 into %arg6[%19]
      : tensor<8x8xf32> into tensor<8x8xf32>[!gml_st.tile<8x8>],
  } : tensor<8x8xf32>, tensor<8x8xf32>
  return %13#0 : tensor<8x8xf32>
}

// CHECK-LABEL: @fold_for_iter_arg
// CHECK:         %[[INIT:.*]] = tensor.empty()
// CHECK-NOT:     tensor.empty()
// CHECK:         %[[FOR:.*]] = gml_st.for {{.*}} outs (%[[ARG:.*]] = %[[INIT]]: tensor<8x8xf32>) {
// CHECK:         gml_st.set_yield {{.*}} into %[[ARG]][{{.*}}] : tensor<8x8xf32> into tensor<8x8xf32>
// CHECK:         } : tensor<8x8xf32>
// CHECK:         return %[[FOR]] : tensor<8x8xf32

// -----

func.func @fold_for_iter_arg_no_args(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %1 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.for (%arg4) = (%c0) to (%c8) step (%c8) outs (%arg6 = %1: tensor<8x8xf32>) {
    %19 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    gml_st.set_yield %arg6 into %arg6[%19]
      : tensor<8x8xf32> into tensor<8x8xf32>[!gml_st.tile<8x8>],
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @fold_for_iter_arg_no_args
// CHECK:         %[[INIT:.*]] = tensor.empty()
// CHECK-NEXT:    return %[[INIT]] : tensor<8x8xf32

// -----

func.func @collapse_empty_for_vector(%in: vector<8x8xf32>) -> vector<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x8xf32>
  %0 = tensor.empty() : tensor<8x8xf32>
  %6 = vector.transfer_read %0[%c0, %c0], %cst {in_bounds = [true, true]} :
        tensor<8x8xf32>, vector<8x8xf32>
  %13 = gml_st.for (%arg4) = (%c0) to (%c1) step (%c8)
        outs (%arg5 = %6: vector<8x8xf32>) {
    %19 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %20 = tensor.extract_slice %0[0, 0] [8, 8] [1, 1]
      : tensor<8x8xf32> to tensor<8x8xf32>
    %7 = vector.transfer_write %arg5, %20[%c0, %c0] {in_bounds = [true, true]} :
          vector<8x8xf32>, tensor<8x8xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%7 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    %8 = vector.transfer_read %11[%c0, %c0], %cst {in_bounds = [true, true]} :
          tensor<8x8xf32>, vector<8x8xf32>
    gml_st.set_yield %8 into %arg5[%19] : vector<8x8xf32>
          into vector<8x8xf32>[!gml_st.tile<8x8>]
  } : vector<8x8xf32>
  return %13 : vector<8x8xf32>
}

// CHECK-LABEL: @collapse_empty_for_vector
// CHECK-NOT:     gml_st.for
// CHECK:         linalg.fill
// CHECK:         %[[READ:.*]] = vector.transfer_read
// CHECK:         return %[[READ]] : vector<8x8xf32>

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
