// RUN: mlir-hlo-opt %s --split-input-file --optimize-vector-transfer | FileCheck %s

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

// CHECK-LABEL: func @optimize_pack_with_transpose(
// CHECK-SAME:      %[[INPUT:.*]]: memref<1024x1024xf32>)

// CHECK:         %[[ALLOC:.*]] = memref.alloc
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[INPUT]]
// CHECK-NOT:     vector.broadcast
// CHECK-NOT:     vector.transpose
// CHECK:         %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]
// CHECK-SAME:    memref<128x1024x8x1xf32> into memref<128x1024x8xf32>
// CHECK:         vector.transfer_write %[[READ]], %[[COLLAPSE]]

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

// CHECK-LABEL: func @optimize_pack(
// CHECK-SAME:      %[[INPUT:.*]]: memref<1024x1024xf32>)

// CHECK:         %[[ALLOC:.*]] = memref.alloc
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[INPUT]]
// CHECK:         %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]
// CHECK-SAME:    memref<128x1024x8x1xf32> into memref<128x1024x8xf32>
// CHECK:         %[[SHAPE_CAST:.*]] = vector.shape_cast %[[READ]]
// CHECK-SAME:    vector<8x1xf32> to vector<8xf32>
// CHECK:         vector.transfer_write %[[SHAPE_CAST]], %[[COLLAPSE]]
