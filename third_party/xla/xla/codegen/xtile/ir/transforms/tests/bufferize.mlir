// RUN: emitters_opt %s -one-shot-bufferize -canonicalize -cse \
// RUN: -split-input-file | FileCheck %s

// CHECK: @extract_strided(%[[SOURCE:.*]]: memref<16xf32>, %[[OFFSET:.*]]: index)
func.func @extract_strided(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index

  // CHECK: %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]]
  // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
  // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

  // CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<8xf32>

  // CHECK: %[[BUFFER_SUBVIEW:.*]] = memref.subview %[[BUFFER]]
  // CHECK-SAME: [0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>

  // CHECK: memref.copy %[[INPUT_SUBVIEW]], %[[BUFFER_SUBVIEW]]
  // CHECK-SAME: : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[1]>>

  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]] restrict writable
  // CHECK-SAME: : memref<8xf32> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][2] : memref<16xf32> -> tensor<8xf32>
  // CHECK: return %[[TILE]] : tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @insert_strided(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<16xf32>,
// CHECK-SAME: %[[OFFSET:.*]]: index)
func.func @insert_strided(%source: tensor<8xf32>, %destination: memref<16xf32>, %tile_id: index) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index

  // CHECK: %[[SOURCE_BUFFER:.*]] = bufferization.to_buffer %[[SOURCE]]
  // CHECK-SAME: : tensor<8xf32> to memref<8xf32, strided<[?], offset: ?>>

  // CHECK: %[[SHIFT:.*]] = arith.subi %[[C15]], %[[OFFSET]] : index
  // CHECK: %[[STRIDED_SHIFT:.*]] = arith.divsi %[[SHIFT]], %[[C2]] : index
  // CHECK: %[[ELEMENTS_TO_END:.*]] = arith.addi %[[STRIDED_SHIFT]], %[[C1]] : index
  // CHECK: %[[SIZE:.*]] = arith.minsi %[[ELEMENTS_TO_END]], %[[C8]] : index

  // CHECK: %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER]][0] [%[[SIZE]]] [1]
  // CHECK-SAME: : memref<8xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

  // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]]
  // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
  // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

  // CHECK: memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]]
  // CHECK-SAME: : memref<?xf32, strided<[?], offset: ?>>
  // CHECK-SAME: to memref<?xf32, strided<[2], offset: ?>>
  xtile.insert %source into %destination[%tile_id][8][2] : tensor<8xf32> -> memref<16xf32>
  return
}
