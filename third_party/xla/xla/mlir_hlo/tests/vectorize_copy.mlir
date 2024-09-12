// RUN: mlir-hlo-opt %s --vectorize-copy --split-input-file | FileCheck %s

func.func @vectorize_copy(%subview: memref<2x2xf32, strided<[16, 1]>>) -> memref<2x2xf32> {
  %alloc = memref.alloc() : memref<2x2xf32>
  memref.copy %subview, %alloc : memref<2x2xf32, strided<[16, 1]>> to memref<2x2xf32>
  return %alloc : memref<2x2xf32>
}

// CHECK-LABEL: func @vectorize_copy

// CHECK-NOT:     memref.copy
// CHECK:         vector.transfer_read
// CHECK:         vector.transfer_write

// -----

func.func @do_not_vectorize_continuous_copy(%arg: memref<10x10xf32>) -> memref<10x10xf32> {
  %alloc_10 = memref.alloc() : memref<10x10xf32>
  memref.copy %arg, %alloc_10 : memref<10x10xf32> to memref<10x10xf32>
  return %alloc_10 : memref<10x10xf32>
}

// CHECK-LABEL: func @do_not_vectorize_continuous_copy

// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         memref.copy

// -----

func.func @tile_to_continuous_memref(%arg: memref<3x512xf32, strided<[768, 1]>>)
    -> (memref<3x512xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x512xf32>
  memref.copy %arg, %alloc : memref<3x512xf32, strided<[768, 1]>> to memref<3x512xf32>
  return %alloc: memref<3x512xf32>
}

// CHECK-LABEL: func @tile_to_continuous_memref
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3
// CHECK:         scf.for %[[I:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NOT:       vector.transfer_read
// CHECK-NOT:       vector.transfer_write
// CHECK:           memref.copy

// -----

func.func @tile_middle_dim_and_vectorize(%arg0: memref<4x8x2xi64, strided<[70, 14, 2]>>)
    -> memref<4x8x2xi64> {
  %alloc = memref.alloc() : memref<4x8x2xi64>
  memref.copy %arg0, %alloc : memref<4x8x2xi64, strided<[70, 14, 2]>> to memref<4x8x2xi64>
  return %alloc : memref<4x8x2xi64>
}

// CHECK-LABEL:  func.func @tile_middle_dim_and_vectorize
// CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:      %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:      %[[C8:.*]] = arith.constant 8 : index
// CHECK:          scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-COUNT-2:    memref.subview {{.*}}[%[[I]], 0, 0] [1, 8, 2] [1, 1, 1]
// CHECK:            scf.for %[[J:.*]] = %[[C0]] to %[[C8]] step %[[C4]]
// CHECK-COUNT-2:      memref.subview {{.*}}[0, %[[J]], 0] [1, 4, 2] [1, 1, 1]
// CHECK:              vector.transfer_read
// CHECK-SAME:           memref<1x4x2xi64, strided<[70, 14, 2], offset: ?>>, vector<1x4x2xi64>
// CHECK:              vector.transfer_write
// CHECK-SAME:           vector<1x4x2xi64>, memref<1x4x2xi64, strided<[16, 2, 1], offset: ?>>

// -----

func.func @vectorize_strided_copy(%arg: memref<1000xi64>) -> memref<500xi64> {
  %subview = memref.subview %arg[0] [500] [2] : memref<1000xi64> to memref<500xi64, strided<[2]>>
  %alloc = memref.alloc() : memref<500xi64>
  memref.copy %subview, %alloc : memref<500xi64, strided<[2]>> to memref<500xi64>
  return %alloc : memref<500xi64>
}

// CHECK-LABEL:  func.func @vectorize_strided_copy
// CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:      %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:      %[[C496:.*]] = arith.constant 496 : index
// CHECK:          memref.subview {{.*}}[0] [500] [2]
// CHECK-SAME:       memref<1000xi64> to memref<500xi64, strided<[2]>>
// CHECK:          scf.for %[[I:.*]] = %[[C0]] to %[[C496]] step %[[C8]]
// CHECK-COUNT-2:    memref.subview {{.*}}[%[[I]]] [8] [1]
// CHECK:            vector.transfer_read
// CHECK-SAME:         memref<8xi64, strided<[2], offset: ?>>, vector<8xi64>
// CHECK:            vector.transfer_write
// CHECK-SAME:         vector<8xi64>, memref<8xi64, strided<[1], offset: ?>>
// CHECK-COUNT-2:  memref.subview {{.*}}[496] [4] [1]
// CHECK:          vector.transfer_read
// CHECK-SAME:       memref<4xi64, strided<[2], offset: 992>>, vector<4xi64>
// CHECK:          vector.transfer_write
// CHECK-SAME:       vector<4xi64>, memref<4xi64, strided<[1], offset: 496>>
