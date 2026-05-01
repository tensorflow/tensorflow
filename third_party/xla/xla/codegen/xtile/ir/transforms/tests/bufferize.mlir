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
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32>) {
    // CHECK: %[[STATIC_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [2]
    // CHECK: %[[ALLOC_0:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: memref.copy %[[STATIC_SUBVIEW]], %[[ALLOC_0]]
    // CHECK: scf.yield %[[ALLOC_0]] : memref<8xf32>
  // CHECK: } else {
    // CHECK: %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: %[[ALLOC_1:.*]] = memref.alloc() : memref<8xf32>

    // CHECK: %[[ALLOC_1_SUBVIEW:.*]] = memref.subview %[[ALLOC_1]]
    // CHECK-SAME: [0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>

    // CHECK: memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_1_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[1]>>
    // CHECK: scf.yield %[[ALLOC_1]] : memref<8xf32>
  // CHECK: }

  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]]
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
  // CHECK: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: scf.if %[[IS_FULL_TILE]] {
    // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[OFFSET]]] [8] [2]
    // CHECK:   memref.copy %[[SOURCE_BUFFER]], %[[DESTINATION_SUBVIEW]]
  // CHECK: } else {
    // CHECK: %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER]][0] [%[[SIZE]]] [1]
    // CHECK-SAME: : memref<8xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

    // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2]
    // CHECK-SAME: : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>

    // CHECK: memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]]
    // CHECK-SAME: : memref<?xf32, strided<[?], offset: ?>>
    // CHECK-SAME: to memref<?xf32, strided<[2], offset: ?>>
  // CHECK: }

  xtile.insert %source into %destination[%tile_id][8][2] : tensor<8xf32> -> memref<16xf32>
  return
}

// -----

// CHECK: @extract_identity(%[[SOURCE:.*]]: memref<16xf32>, %[[OFFSET:.*]]: index)
func.func @extract_identity(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[DIFF:.*]] = arith.subi %[[C16]], %[[OFFSET]] : index
  // CHECK-DAG: %[[SIZE:.*]] = arith.minsi %[[DIFF]], %[[C8]] : index
  // CHECK-DAG: %[[IS_FULL_TILE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C8]] : index

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_IF:.*]] = memref.alloc() : memref<8xf32>
  // CHECK:   memref.copy %[[SUBVIEW]], %[[ALLOC_IF]] : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32>
  // CHECK:   scf.yield %[[ALLOC_IF]] : memref<8xf32>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [1] : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  // CHECK:   scf.yield %[[ALLOC_ELSE]] : memref<8xf32>
  // CHECK: }
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]] : memref<8xf32> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @extract_5d(%[[SOURCE:.*]]: memref<1x2x1x32768x256xf32>, %[[D1:.*]]: index, %[[D3:.*]]: index, %[[D4:.*]]: index)
func.func @extract_5d(%source: memref<1x2x1x32768x256xf32>, %d1: index, %d3: index, %d4: index) -> tensor<1x1x1x1x16xf32> {
  // CHECK: %[[C_0:.*]] = arith.constant 0 : index
  // CHECK: %[[BUFFER:.*]] = scf.if {{.*}} -> (memref<1x1x1x1x16xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[C_0]], %[[D1]], %[[C_0]], %[[D3]], %[[D4]]] [1, 1, 1, 1, 16] [0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> to memref<1x1x1x1x16xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[ALLOC_IF:.*]] = memref.alloc() : memref<1x1x1x1x16xf32>
  // CHECK:   memref.copy %[[SUBVIEW]], %[[ALLOC_IF]]
  // CHECK:   scf.yield %[[ALLOC_IF]] : memref<1x1x1x1x16xf32>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %{{.*}} : memref<1x2x1x32768x256xf32> to memref<?x?x?x?x?xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<1x1x1x1x16xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0, 0, 0, 0, 0] [1, %{{.*}}, 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1, 1] : memref<1x1x1x1x16xf32> to memref<1x?x1x?x?xf32, strided<[16, 16, 16, 16, 1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]]
  // CHECK:   scf.yield %[[ALLOC_ELSE]] : memref<1x1x1x1x16xf32>
  // CHECK: }

  // CHECK: bufferization.to_tensor %[[BUFFER]] : memref<1x1x1x1x16xf32> to tensor<1x1x1x1x16xf32>
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0, %d1, %c0, %d3, %d4][1, 1, 1, 1, 16][0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> -> tensor<1x1x1x1x16xf32>
  return %tile : tensor<1x1x1x1x16xf32>
}

// -----

// CHECK: @extract_static(%[[SOURCE:.*]]: memref<16xf32>)
func.func @extract_static(%source: memref<16xf32>) -> tensor<8xf32> {
  // CHECK-NOT: scf.if
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][0] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: memref.copy %[[SUBVIEW]], %[[ALLOC]] : memref<8xf32, strided<[1]>> to memref<8xf32>
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[ALLOC]] : memref<8xf32> to tensor<8xf32>
  // CHECK: return %[[TILE]]
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}
