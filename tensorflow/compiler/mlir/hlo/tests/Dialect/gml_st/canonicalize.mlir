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
      iterators["reduction"] {
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
      iterators["reduction", "parallel"] {
    %in_sub = tensor.extract_slice %in_[%i][4][1]
      : tensor<16xf32> to tensor<4xf32>
    %result = func.call @reduce(%in_sub, %out_):
      (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    gml_st.yield %result : tensor<f32>
  }
  func.return %0 : tensor<f32>
}
