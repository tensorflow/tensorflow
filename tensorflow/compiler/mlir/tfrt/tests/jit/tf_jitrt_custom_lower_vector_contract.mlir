// RUN: tf-tfrt-opt %s -lowering-vector-contract | FileCheck %s

func.func @lower_vector_contract(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>)
                  -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %2 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %4 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %3, %4 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  %6 = vector.transfer_write %5, %0[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// CHECK-LABEL: func @lower_vector_contract(
// CHECK-SAME:      %[[LHS:.*]]: tensor<8x8xf32>, %[[RHS:.*]]: tensor<8x8xf32>)

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[INIT:.*]] = tensor.empty

// CHECK:         %[[LHS_READ:.*]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_READ:.*]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[OUT_READ:.*]] = vector.transfer_read %[[INIT]]
// CHECK:         %[[TRANSPOSE:.*]] = vector.transpose %[[LHS_READ]]
// CHECK:         %[[EXTRACT_LHS0:.*]] = vector.extract %[[TRANSPOSE]][0]
// CHECK:         %[[EXTRACT_RHS0:.*]] = vector.extract %[[RHS_READ]][0]
// CHECK:         %[[PRODUCT0:.*]] = vector.outerproduct %[[EXTRACT_LHS0]], %[[EXTRACT_RHS0]], %[[OUT_READ]]
// CHECK:         %[[EXTRACT_LHS1:.*]] = vector.extract %[[TRANSPOSE]][1]
// CHECK:         %[[EXTRACT_RHS1:.*]] = vector.extract %[[RHS_READ]][1]
// CHECK:         %[[PRODUCT1:.*]] = vector.outerproduct %[[EXTRACT_LHS1]], %[[EXTRACT_RHS1]], %[[PRODUCT0]]
// CHECK:         %[[PRODUCT2:.*]] = vector.outerproduct %[[EXTRACT_LHS2:.*]], %[[EXTRACT_RHS2:.*]], %[[PRODUCT1]]
// CHECK:         %[[PRODUCT3:.*]] = vector.outerproduct %[[EXTRACT_LHS3:.*]], %[[EXTRACT_RHS3:.*]], %[[PRODUCT2]]
// CHECK:         %[[PRODUCT4:.*]] = vector.outerproduct %[[EXTRACT_LHS4:.*]], %[[EXTRACT_RHS4:.*]], %[[PRODUCT3]]
// CHECK:         %[[PRODUCT5:.*]] = vector.outerproduct %[[EXTRACT_LHS5:.*]], %[[EXTRACT_RHS5:.*]], %[[PRODUCT4]]
// CHECK:         %[[PRODUCT6:.*]] = vector.outerproduct %[[EXTRACT_LHS6:.*]], %[[EXTRACT_RHS6:.*]], %[[PRODUCT5]]
// CHECK:         %[[PRODUCT7:.*]] = vector.outerproduct %[[EXTRACT_LHS7:.*]], %[[EXTRACT_RHS7:.*]], %[[PRODUCT6]]
// CHECK:         %[[RET:.*]] = vector.transfer_write %[[PRODUCT7]], %[[INIT]]
// CHECK:         return %[[RET]]

// -----

func.func @canonicalize_outer_product(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>)
                  -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x8xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %2 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %4 = gml_st.tile [0, 0] [8, 8] [1, 1] : !gml_st.tile<8x8>
  %5 = gml_st.materialize %cst[%4] : vector<8x8xf32>[!gml_st.tile<8x8>] to vector<8x8xf32>
  %6 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %3, %5 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  %7 = vector.transfer_write %6, %0[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
  return %7 : tensor<8x8xf32>
}

// CHECK-LABEL: func @canonicalize_outer_product(
// CHECK-SAME:      %[[LHS:.*]]: tensor<8x8xf32>, %[[RHS:.*]]: tensor<8x8xf32>)

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
// CHECK:         %[[INIT:.*]] = tensor.empty

// CHECK:         %[[LHS_READ:.*]] = vector.transfer_read %[[LHS]]
// CHECK:         %[[RHS_READ:.*]] = vector.transfer_read %[[RHS]]
// CHECK:         %[[TRANSPOSE:.*]] = vector.transpose %[[LHS_READ]]
// CHECK:         %[[EXTRACT_LHS0:.*]] = vector.extract %[[TRANSPOSE]][0]
// CHECK:         %[[EXTRACT_RHS0:.*]] = vector.extract %[[RHS_READ]][0]
// CHECK:         %[[PRODUCT0:.*]] = vector.outerproduct %[[EXTRACT_LHS0]], %[[EXTRACT_RHS0]], %[[CST]]
