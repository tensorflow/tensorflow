// RUN: mlir-hlo-opt %s -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @memref_cast_into_loop(
func.func @memref_cast_into_loop(%arg0: memref<192xf32>)  {
  %0 = memref.cast %arg0
    : memref<192xf32> to memref<192xf32, #map>
  %cst = arith.constant 0.000000e+00 : f32
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  // CHECK: gml_st.loop
  // CHECK-SAME: outs (%{{.*}} = %{{.*}}: memref<192xf32>)
  gml_st.loop (%arg3) = (%c0) to (%c192) step (%c24)
    outs (%out = %0: memref<192xf32, #map>) {
    %14 = affine.min affine_map<(d0) -> (-d0 + 192, 24)>(%arg3)
    %16 = memref.subview %out[%arg3] [%14] [1]
      : memref<192xf32, #map> to memref<?xf32, #map>
    linalg.fill ins(%cst : f32) outs(%16 : memref<?xf32, #map>)
    gml_st.yield
  }
  func.return
}

// -----

func.func private @foo(%A: memref<48xf32>, %B: tensor<48xf32>,
                  %C: memref<48xf32>) -> (tensor<48xf32>)

func.func @fold_loop_results(%A: memref<48xf32>, %B: tensor<48xf32>,
    %C: memref<48xf32>, %C_tensor: tensor<48xf32>) -> tensor<48xf32> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c48 = arith.constant 48 : index
  %useful, %useless = gml_st.loop (%i) = (%c0) to (%c48) step (%c24)
      ins (%A_ = %A: memref<48xf32>)
      outs (%B_ = %B: tensor<48xf32>,
            %CT_ = %C_tensor: tensor<48xf32>,
            %C_ = %C: memref<48xf32>) {
        %result = func.call @foo(%A_, %B_, %C_)
          : (memref<48xf32>, tensor<48xf32>, memref<48xf32>)-> (tensor<48xf32>)
    gml_st.yield %result, %CT_ : tensor<48xf32>, tensor<48xf32>
  }
  func.return %useful : tensor<48xf32>
}

// CHECK-LABEL: func @fold_loop_results(
// CHECK-SAME:   %[[A:.*]]: [[BUF_TY:memref<48xf32>]], %[[B:.*]]: [[TY:tensor<48xf32>]],
// CHECK-SAME:   %[[C:.*]]: [[BUF_TY]],  %[[C_TENSOR:.*]]: [[TY]]) -> [[TY]] {

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:  %[[C48:.*]] = arith.constant 48 : index

// CHECK-NOT: %{{.*}} = gml_st.loop
// CHECK:  %[[RESULT:.*]] = gml_st.loop (%{{.*}}) = (%[[C0]])
// CHECK-SAME: to (%[[C48]]) step (%[[C24]])
// CHECK-SAME: ins (%[[A_:.*]] = %[[A]]: [[BUF_TY]])
// CHECK-SAME: outs (%[[B_:.*]] = %[[B]]: [[TY]], %[[C_:.*]] = %[[C]]: [[BUF_TY]]) {
// CHECK-NEXT:   %[[RES:.*]] = func.call @foo(%[[A_]], %[[B_]], %[[C_]])
// CHECK-NEXT:   gml_st.yield %[[RES]] :

// CHECK: return %[[RESULT]]

// -----

func.func private @foo(%A: memref<192xf32>, %B: tensor<192xf32>) -> tensor<192xf32>

func.func @fold_loop_inputs(%A: memref<192xf32>, %A_tensor: tensor<192xf32>,
                             %B_tensor: tensor<192xf32>) -> tensor<192xf32> {
  %c0 = arith.constant 0 : index
  %c24 = arith.constant 24 : index
  %c192 = arith.constant 192 : index
  %result = gml_st.loop (%i) = (%c0) to (%c192) step (%c24)
      ins (%A_ = %A: memref<192xf32>, %AT_ = %A_tensor: tensor<192xf32>)
      outs (%BT_ = %B_tensor: tensor<192xf32>) {
    %0 = func.call @foo(%A_, %BT_) : (memref<192xf32>, tensor<192xf32>) -> tensor<192xf32>
    gml_st.yield %0 : tensor<192xf32>
  }
  func.return %result : tensor<192xf32>
}

// CHECK-LABEL: func @fold_loop_inputs
// CHECK: %[[RESULT:.*]] = gml_st.loop
// CHECK-SAME: ins (%{{.*}} = %{{.*}}: memref<192xf32>)

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dim_of_loop_input_no_canonicalize(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   gml_st.loop {{.*}} outs (%[[o:.*]] =
//       CHECK:     %[[dim:.*]] = tensor.dim %[[o]], %[[c0]]
//       CHECK:     arith.index_cast %[[dim]]
func.func @dim_of_loop_input_no_canonicalize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = gml_st.loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %inner_dim = tensor.dim %out1, %c0 : tensor<?x?xf32>
    %cast1 = arith.index_cast %inner_dim : index to i32
    %cast2 = arith.sitofp %cast1 : i32 to f32
    %fill = linalg.fill ins(%cast2 : f32) outs(%out1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %slice = tensor.extract_slice %fill[0, 0][%s, %s][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    gml_st.yield %slice : tensor<?x?xf32>
  }
  func.return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dim_of_loop_input(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   gml_st.loop
//       CHECK:     %[[dim:.*]] = tensor.dim %[[arg1]], %[[c0]]
//       CHECK:     arith.index_cast %[[dim]]
func.func @dim_of_loop_input(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = gml_st.loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %inner_dim = tensor.dim %in1, %c0 : tensor<?x?xf32>
    %cast1 = arith.index_cast %inner_dim : index to i32
    %cast2 = arith.sitofp %cast1 : i32 to f32
    %fill = linalg.fill ins(%cast2 : f32) outs(%out1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    gml_st.yield %fill : tensor<?x?xf32>
  }
  func.return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @dim_of_loop_result(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   tensor.dim %[[arg2]], %[[c0]]
func.func @dim_of_loop_result(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = gml_st.loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %1 = tensor.insert_slice %arg0 into %out1 [0, 0] [%s, %s] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    gml_st.yield %1 : tensor<?x?xf32>
  }
  %r2 = tensor.dim %r, %c0 : tensor<?x?xf32>
  func.return %r2 : index
}

// -----

// CHECK-LABEL: func @dim_of_loop_result_no_canonicalize(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>, %[[arg2:.*]]: tensor<?x?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[r:.*]] = gml_st.loop
//       CHECK:   tensor.dim %[[r]], %[[c0]]
func.func @dim_of_loop_result_no_canonicalize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %s: index)
    -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = gml_st.loop (%iv0, %iv1) = (%c0, %c0)
      to (%d0, %d1) step (%c1, %c1)
      ins (%in0 = %arg0 : tensor<?x?xf32>, %in1 = %arg1 : tensor<?x?xf32>)
      outs (%out1 = %arg2 : tensor<?x?xf32>) {
    %1 = tensor.insert_slice %arg0 into %arg1 [0, 0] [%s, %s] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    gml_st.yield %1 : tensor<?x?xf32>
  }
  %r2 = tensor.dim %r, %c0 : tensor<?x?xf32>
  func.return %r2 : index
}

// -----

func.func private @do(%A: tensor<?x4xf32>, %B: tensor<?xf32>) -> tensor<?xf32>

func.func @fold_tensor_cast(%in: tensor<4x600xf32>,
                       %out: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c600 = arith.constant 600 : index

  %in_cast = tensor.cast %in : tensor<4x600xf32> to tensor<?x600xf32>
  %out_cast = tensor.cast %out : tensor<4xf32> to tensor<?xf32>

  %result = gml_st.loop (%i) = (%c0) to (%c600) step (%c4)
      ins (%in_ = %in_cast: tensor<?x600xf32>)
      outs (%out_ = %out_cast: tensor<?xf32>)
      iterators[#gml_st.iterator_type<reduction>] {
    %dim_in = tensor.dim %in_, %c0 : tensor<?x600xf32>
    %dim_out = tensor.dim %out_, %c0 : tensor<?xf32>

    %in_sub = tensor.extract_slice %in_[0, %i] [%dim_in, 4] [1, 1]
      : tensor<?x600xf32> to tensor<?x4xf32>
    %out_sub = tensor.extract_slice %out_[0] [%dim_out] [1]
      : tensor<?xf32> to tensor<?xf32>
    %result_sub = func.call @do(%in_sub, %out_sub):
      (tensor<?x4xf32>, tensor<?xf32>) -> tensor<?xf32>
    %out_update = tensor.insert_slice %result_sub into %out_[0] [%dim_out] [1]
      : tensor<?xf32> into tensor<?xf32>
    gml_st.yield %out_update : tensor<?xf32>
  }
  %result_cast = tensor.cast %result : tensor<?xf32> to tensor<4xf32>
  func.return %result_cast : tensor<4xf32>
}

// CHECK-LABEL: func @fold_tensor_cast(
// CHECK-SAME:    %[[IN:.*]]: tensor<4x600xf32>, %[[OUT:.*]]: tensor<4xf32>)

// CHECK-DAG:  %[[C600:.*]] = arith.constant 600 : index
// CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index

// CHECK:      %[[RESULT:.*]] = gml_st.loop
// CHECK-SAME:   ins (%[[IN_:.*]] = %[[IN]]: tensor<4x600xf32>)
// CHECK-SAME:   outs (%[[OUT_:.*]] = %[[OUT]]: tensor<4xf32>) iterators

// CHECK:      %[[IN_SUB:.*]] = tensor.extract_slice
// CHECK:      %[[IN_SUB_CAST:.*]] = tensor.cast %[[IN_SUB]]
// CHECK-SAME:   : tensor<4x4xf32> to tensor<?x4xf32>

// CHECK:      %[[OUT_SUB:.*]] = tensor.cast %[[OUT_]]
// CHECK-SAME:   : tensor<4xf32> to tensor<?xf32>

// CHECK:      %[[RESULT_SUB:.*]] = func.call @do(%[[IN_SUB_CAST]], %[[OUT_SUB]])
// CHECK:      %[[RESULT_CAST:.*]] = tensor.cast %[[RESULT_SUB]]
// CHECK:      gml_st.yield %[[RESULT_CAST]] : tensor<4xf32>
// CHECK:    }
// CHECK:    return %[[RESULT]] : tensor<4xf32>

// -----

func.func private @reduce(%A: tensor<4xf32>, %B: tensor<f32>) -> tensor<f32>

// CHECK-LABEL: @remove_empty_loop
func.func @remove_empty_loop(%in: tensor<16xf32>, %out: tensor<f32>,
                             %buf: memref<f32>) -> tensor<f32>{
  // CHECK-NOT: gml_st.loop
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %0 = gml_st.loop (%i, %j) = (%c0, %c0) to (%c16, %c0) step (%c4, %c4)
      ins (%in_ = %in: tensor<16xf32>)
      outs (%out_ = %out: tensor<f32>, %buf_ = %buf: memref<f32>)
      iterators[#gml_st.iterator_type<reduction>,
                #gml_st.iterator_type<parallel>] {
    %in_sub = tensor.extract_slice %in_[%i][4][1]
      : tensor<16xf32> to tensor<4xf32>
    %result = func.call @reduce(%in_sub, %out_):
      (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    gml_st.yield %result : tensor<f32>
  }
  func.return %0 : tensor<f32>
}

// -----

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

// CHECK-LABEL: @fold_constant_tile_through_materialize
func.func @fold_constant_tile_through_materialize(%in: tensor<4xf32>) ->
    tensor<?xf32> {
  %c2 = arith.constant 2 : index
  // CHECK: %[[TILE:.*]] = gml_st.tile [2] [2] [1] : !gml_st.tile<2>
  %tile = gml_st.tile [%c2] [%c2] [1] : !gml_st.tile<?>
  // CHECK: %[[MAT:.*]] = gml_st.materialize {{.*}}[%[[TILE]]] : tensor<4xf32>[!gml_st.tile<2>]
  %mat = gml_st.materialize %in[%tile] : tensor<4xf32>[!gml_st.tile<?>]
      to tensor<?xf32>
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
    %tile = gml_st.tile [0, 0] [%c2, %c2] [1, 1] : !gml_st.tile<?x?>
    %out_sub = gml_st.materialize %out[%tile] :
                    tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32)
                        outs(%out_sub : tensor<?x?xf32>) -> tensor<?x?xf32>
    gml_st.set_yield %fill into %arg1[%tile] :
                    tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @fold_constant_set_yield
// CHECK:         %[[FOR:.*]] = gml_st.for{{.*}}: tensor<?x?xf32>
// CHECK:           %[[TILE:.*]] = gml_st.tile [0, 0] [2, 2] {{.*}} !gml_st.tile<2x2>
// CHECK-NOT:       builtin.unrealized_conversion_cast
// CHECK-NEXT:      %[[SLICE:.*]] = gml_st.materialize %{{.*}}[%[[TILE]]] {{.*}} to tensor<2x2xf32>
// CHECK:           %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[SLICE]] : tensor<2x2xf32>)
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
// CHECK:         %[[FOR:.*]] = gml_st.for{{.*}}: tensor<?xf32>
// CHECK:           %[[TILE:.*]] = gml_st.tile [0] [1] {{.*}} !gml_st.tile<1>
// CHECK-NOT:       builtin.unrealized_conversion_cast
// CHECK-NEXT:      gml_st.set_yield %[[SCALAR:.*]] into %{{.*}}[%[[TILE]]] : f32 into tensor<?xf32>[!gml_st.tile<1>]

// -----

func.func @fold_constant_for(%in: tensor<?x?xf32>,
                             %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = gml_st.tile [0, 0] [8, 2] [1, 1] : !gml_st.tile<8x2>
  %3 = gml_st.materialize %out[%1] :
                  tensor<?x?xf32>[!gml_st.tile<8x2>] to tensor<8x2xf32>
  %cast_3 = tensor.cast %3 : tensor<8x2xf32> to tensor<?x?xf32>
  %4 = gml_st.for (%arg0) = (%c0) to (%c8) step (%c2)
                  outs (%arg1 = %cast_3: tensor<?x?xf32>) {
    %tile = gml_st.tile [0, %arg0] [8, 2] [1, 1] : !gml_st.tile<8x2>
    %2 = builtin.unrealized_conversion_cast %tile :
                    !gml_st.tile<8x2> to !gml_st.tile<?x?>
    %out_sub = gml_st.materialize %arg1[%tile] :
                    tensor<?x?xf32>[!gml_st.tile<8x2>] to tensor<8x2xf32>
    %fill = linalg.fill ins(%cst : f32)
                        outs(%out_sub : tensor<8x2xf32>) -> tensor<8x2xf32>
    %cast_fill = tensor.cast %fill : tensor<8x2xf32> to tensor<?x?xf32>
    gml_st.set_yield %cast_fill into %arg1[%2] :
                    tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// CHECK-LABEL: @fold_constant_for
// CHECK:         %[[SLICE:.*]] = gml_st.materialize {{.*}} to tensor<8x2xf32>
// CHECK-NOT:     tensor.cast
// CHECK:         %[[FOR1:.*]] = gml_st.for (%{{.*}} = (%c0) to {{.*}} outs (%[[ARG1:.*]] = %[[SLICE]]: tensor<8x2xf32>
// CHECK:           %[[FOR1_TILE:.*]] = gml_st.tile {{.*}} [8, 2] {{.*}} !gml_st.tile<8x2>
// CHECK-NEXT:      %[[FOR1_SLICE:.*]] = gml_st.materialize %{{.*}}[%[[FOR1_TILE]]] {{.*}} to tensor<8x2xf32>
// CHECK:           gml_st.set_yield %{{.*}} into %[[ARG1]][%[[FOR1_TILE]]] : tensor<8x2xf32> into tensor<8x2xf32>[!gml_st.tile<8x2>]
// CHECK-NEXT:    } : tensor<8x2xf32>
// CHECK:         %[[CAST:.*]] = tensor.cast %[[FOR1]] : tensor<8x2xf32> to tensor<?x?xf32>
// CHECK-NEXT:    return %[[CAST]] : tensor<?x?xf32>

// -----

func.func @fold_cast_to_materialize_source(%in: tensor<4xf32>) ->
    tensor<2xf32> {
  %tile = gml_st.tile [2] [2] [1] : !gml_st.tile<2>
  %cast = tensor.cast %in : tensor<4xf32> to tensor<?xf32>
  %mat = gml_st.materialize %cast[%tile] : tensor<?xf32>[!gml_st.tile<2>]
      to tensor<2xf32>
  func.return %mat : tensor<2xf32>
}

// CHECK-LABEL: @fold_cast_to_materialize_source
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>
// CHECK:         %[[TILE:.*]] = gml_st.tile [2] [2] [1] : !gml_st.tile<2>
// CHECK-NOT:     tensor.cast
// CHECK:         %[[MAT:.*]] = gml_st.materialize %[[IN]][%[[TILE]]] : tensor<4xf32>[!gml_st.tile<2>]
// CHECK:         return %[[MAT]]

// -----

func.func @collapse_empty_for(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
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

// CHECK-LABEL: @collapse_empty_for
// CHECK-NOT:     gml_st.for
// CHECK:         linalg.fill

// -----

func.func @collapse_empty_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c0) to (%c1, %c1)
        step (%c8, %c8) {
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %20 = gml_st.materialize %0[%19] : tensor<8x8xf32>[!gml_st.tile<8x8>] to tensor<8x8xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%20 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %0[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>]
  } : tensor<8x8xf32>
  return %13 : tensor<8x8xf32>
}

// CHECK-LABEL: @collapse_empty_parallel
// CHECK-NOT:     gml_st.parallel
// CHECK:         gml_st.tile [0, 0]
// CHECK:         linalg.fill

// -----

func.func @collapse_one_dim_parallel(%in: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %13 = gml_st.parallel (%arg4, %arg5) = (%c0, %c0) to (%c1, %c16)
        step (%c8, %c8) {
    %19 = gml_st.tile [%arg4, %arg5] [8, 8] [1, 1] : !gml_st.tile<8x8>
    %11 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8x8xf32>)
          -> tensor<8x8xf32>
    gml_st.set_yield %11 into %0[%19] : tensor<8x8xf32>
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
    gml_st.set_yield %11 into %arg5[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>],
          %arg6 into %arg6[%19] : tensor<8x8xf32>
          into tensor<8x8xf32>[!gml_st.tile<8x8>],
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
    %20 = gml_st.materialize %0[%19] : tensor<8x8xf32>[!gml_st.tile<8x8>] to
          tensor<8x8xf32>
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
