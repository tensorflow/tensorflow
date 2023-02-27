// RUN: mlir-hlo-opt %s --lower-vectors --split-input-file | FileCheck %s

// CHECK-LABEL: func @vector_row
func.func @vector_row(%arg0: vector<2x4xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [1] : vector<2x4xf32> to vector<2xf32>
    func.return %0 : vector<2xf32>
}
// CHECK-COUNT-4: arith.mulf

// -----

// CHECK-LABEL: func @vector_col
func.func @vector_col(%arg0: vector<2x4xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<2x4xf32> to vector<4xf32>
    func.return %0 : vector<4xf32>
}
// CHECK: arith.mulf
// CHECK: arith.mulf

// -----

// CHECK-LABEL: func @vector_1d
func.func @vector_1d(%arg0: vector<4xf32>, %acc: f32) -> f32 {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [0] : vector<4xf32> to f32
    func.return %0 : f32
}

// -----

// CHECK: vector.reduction <mul>
func.func @lower_vector_contract(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>)
                  -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %2 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %4 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<8x8xf32>, vector<8x8xf32>
  %5 = vector.contract {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d2, d1)>,
                          affine_map<(d0, d1, d2) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel", "reduction"],
         kind = #vector.kind<add>
  } %2, %3, %4 : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
  %6 = vector.transfer_write %5, %0[%c0, %c0] {in_bounds = [true, true]} : vector<8x8xf32>, tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @lower_vector_contract
// CHECK-COUNT-8: vector.outerproduct

// -----

func.func @lower_vector_contract_4d(%arg0: tensor<1x1x8x1xf32>,
                                    %arg1: tensor<1x1x8x1xf32>)
                  -> tensor<1x1x8x8xf32> {
  %c0 = arith.constant 0 : index
  %4 = tensor.empty() : tensor<1x1x8x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %20 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x8x1xf32>,
                                             vector<1x1x8x1xf32>
  %21 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x8x1xf32>,
                                             vector<1x1x8x1xf32>
  %22 = vector.transfer_read %4[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x8x8xf32>,
                                             vector<1x1x8x8xf32>
  %23 = vector.contract {indexing_maps =
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %20, %21, %22 : vector<1x1x8x1xf32>, vector<1x1x8x1xf32>
               into vector<1x1x8x8xf32>
  %14 = vector.transfer_write %23, %4[%c0, %c0, %c0, %c0]
    {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>,
                                             tensor<1x1x8x8xf32>
  return %14 : tensor<1x1x8x8xf32>
}

// CHECK-LABEL: func @lower_vector_contract_4d
// CHECK:         vector.outerproduct

// -----

func.func @lower_vector_contract_4d_matvec(%arg0: tensor<1x1x1x1xf32>,
                                           %arg1: tensor<1x1x8x1xf32>)
                  -> tensor<1x1x1x8xf32> {
  %c0 = arith.constant 0 : index
  %4 = tensor.empty() : tensor<1x1x1x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %20 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x1x1xf32>,
                                             vector<1x1x1x1xf32>
  %21 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x8x1xf32>,
                                             vector<1x1x8x1xf32>
  %22 = vector.transfer_read %4[%c0, %c0, %c0, %c0], %cst
    {in_bounds = [true, true, true, true]} : tensor<1x1x1x8xf32>,
                                             vector<1x1x1x8xf32>
  %23 = vector.contract {indexing_maps =
    [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>],
    iterator_types = ["parallel", "parallel", "reduction",
                      "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %20, %21, %22 : vector<1x1x1x1xf32>, vector<1x1x8x1xf32>
               into vector<1x1x1x8xf32>
  %14 = vector.transfer_write %23, %4[%c0, %c0, %c0, %c0]
    {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf32>,
                                             tensor<1x1x1x8xf32>
  return %14 : tensor<1x1x1x8xf32>
}

// CHECK-LABEL: func @lower_vector_contract_4d_matvec
// CHECK:         vector.outerproduct
