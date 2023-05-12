// RUN: mlir-hlo-opt %s --lower-vectors --split-input-file | FileCheck %s
// RUN: mlir-hlo-opt %s --lower-vectors="flatten=true" --split-input-file | FileCheck %s --check-prefix=FLATTEN

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
  %2 = vector.transfer_read %arg0[%c0, %c0], %cst_0 {in_bounds = [true, true]}
    : tensor<8x8xf32>, vector<8x8xf32>
  %3 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true]}
    : tensor<8x8xf32>, vector<8x8xf32>
  %4 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]}
    : tensor<8x8xf32>, vector<8x8xf32>
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

// -----

#map = affine_map<(d0) -> (d0 * 8)>
func.func @optimize_pack_with_transpose(%arg0: memref<1024x1024xf32>) ->
                                        memref<128x1024x8x1xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x1024x8x1xf32>
  scf.for %arg2 = %c0 to %c128 step %c1 {
    scf.for %arg3 = %c0 to %c1024 step %c1 {
      %0 = affine.apply #map(%arg2)
      %1 = vector.transfer_read %arg0[%arg3, %0], %cst {in_bounds = [true]} :
                                memref<1024x1024xf32>, vector<8xf32>
      %2 = vector.broadcast %1 : vector<8xf32> to vector<1x8xf32>
      %3 = vector.transpose %2, [1, 0] : vector<1x8xf32> to vector<8x1xf32>
      vector.transfer_write %3, %alloc_0[%arg2, %arg3, %c0, %c0]
                            {in_bounds = [true, true]} :
                            vector<8x1xf32>, memref<128x1024x8x1xf32>
    }
  }
  return %alloc_0 : memref<128x1024x8x1xf32>
}

// FLATTEN-LABEL: func @optimize_pack_with_transpose(
// FLATTEN-SAME:      %[[INPUT:.*]]: memref<1024x1024xf32>)

// FLATTEN:         %[[ALLOC:.*]] = memref.alloc
// FLATTEN:         %[[READ:.*]] = vector.transfer_read %[[INPUT]]
// FLATTEN-NOT:     vector.broadcast
// FLATTEN-NOT:     vector.transpose
// FLATTEN:         %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]
// FLATTEN-SAME:    memref<128x1024x8x1xf32> into memref<128x1024x8xf32>
// FLATTEN:         %[[SHAPE_CAST:.*]] = vector.shape_cast %{{.*}}
// FLATTEN:         vector.transfer_write %[[SHAPE_CAST]], %[[COLLAPSE]]

// -----

#map = affine_map<(d0) -> (d0 * 8)>
func.func @optimize_pack(%arg0: memref<1024x1024xf32>) ->
                         memref<128x1024x8x1xf32> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x1024x8x1xf32>
  scf.for %arg2 = %c0 to %c128 step %c1 {
    scf.for %arg3 = %c0 to %c1024 step %c1 {
      %0 = affine.apply #map(%arg2)
      %1 = vector.transfer_read %arg0[%0, %arg3], %cst
                                {in_bounds = [true, true]} :
                                memref<1024x1024xf32>, vector<8x1xf32>
      vector.transfer_write %1, %alloc_0[%arg2, %arg3, %c0, %c0]
                            {in_bounds = [true, true]} :
                            vector<8x1xf32>, memref<128x1024x8x1xf32>
    }
  }
  return %alloc_0 : memref<128x1024x8x1xf32>
}

// FLATTEN-LABEL: func @optimize_pack(
// FLATTEN-SAME:      %[[INPUT:.*]]: memref<1024x1024xf32>)

// FLATTEN:         %[[ALLOC:.*]] = memref.alloc
// FLATTEN:         %[[READ:.*]] = vector.transfer_read %[[INPUT]]
// FLATTEN:         %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]
// FLATTEN-SAME:    memref<128x1024x8x1xf32> into memref<128x1024x8xf32>
// FLATTEN:         %[[SHAPE_CAST:.*]] = vector.shape_cast
// FLATTEN-SAME:    vector<8x1xf32> to vector<8xf32>
// FLATTEN:         vector.transfer_write %[[SHAPE_CAST]], %[[COLLAPSE]]

// -----

func.func @no_flatten(%arg0: memref<2x9x10x2xf64>) ->
                         memref<2x9x10x2xf64> {
  %cst = arith.constant 0.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %alloca = memref.alloca() : memref<2x9x10x2xf64>
  %1 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x9x10x2xf64>, vector<2x9x10x2xf64>
  vector.transfer_write %1, %alloca[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<2x9x10x2xf64>, memref<2x9x10x2xf64>
  return %alloca : memref<2x9x10x2xf64>
}


// CHECK-LABEL:     func @no_flatten(

// CHECK-NOT:         memref.collapse_shape
// CHECK-COUNT-180:   vector.transfer_read {{.*}} memref<2x9x10x2xf64>, vector<2xf64>
// CHECK-COUNT-180:   vector.transfer_write {{.*}} vector<2xf64>, memref<2x9x10x2xf64>
