// RUN: emitters_opt %s --allow-unregistered-dialect -one-shot-bufferize="allow-unknown-ops=true" -canonicalize -cse \
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

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32, strided<[1], offset: ?>>) {
    // CHECK: %[[STATIC_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [2] : memref<16xf32> to memref<8xf32, strided<[2], offset: ?>>
    // CHECK: %[[ALLOC_THEN:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: memref.copy %[[STATIC_SUBVIEW]], %[[ALLOC_THEN]] : memref<8xf32, strided<[2], offset: ?>> to memref<8xf32>
    // CHECK: %[[CAST_0:.*]] = memref.cast %[[ALLOC_THEN]] : memref<8xf32> to memref<8xf32, strided<[1], offset: ?>>
    // CHECK: scf.yield %[[CAST_0]] : memref<8xf32, strided<[1], offset: ?>>
  // CHECK: } else {
    // CHECK: %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [2] : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>
    // CHECK: %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xf32>
    // CHECK: %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
    // CHECK: memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]] : memref<?xf32, strided<[2], offset: ?>> to memref<?xf32, strided<[1]>>
    // CHECK: %[[CAST_1:.*]] = memref.cast %[[ALLOC_ELSE]] : memref<8xf32> to memref<8xf32, strided<[1], offset: ?>>
    // CHECK: scf.yield %[[CAST_1]] : memref<8xf32, strided<[1], offset: ?>>
  // CHECK: }
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]] restrict
  // CHECK-SAME: : memref<8xf32, strided<[1], offset: ?>> to tensor<8xf32>

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
    // CHECK:   %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]][%[[OFFSET]]] [8] [2] : memref<16xf32> to memref<8xf32, strided<[2], offset: ?>>
    // CHECK:   memref.copy %[[SOURCE_BUFFER:.*]], %[[DESTINATION_SUBVIEW]] : memref<8xf32, strided<[?], offset: ?>> to memref<8xf32, strided<[2], offset: ?>>
  // CHECK: } else {
    // CHECK: %[[SOURCE_SUBVIEW:.*]] = memref.subview %[[SOURCE_BUFFER:.*]][0] [%[[SIZE:.*]]] [1] : memref<8xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>
    // CHECK: %[[DESTINATION_SUBVIEW:.*]] = memref.subview %[[DESTINATION]]
    // CHECK-SAME: [%[[OFFSET]]] [%[[SIZE]]] [2] : memref<16xf32> to memref<?xf32, strided<[2], offset: ?>>
    // CHECK: memref.copy %[[SOURCE_SUBVIEW]], %[[DESTINATION_SUBVIEW]] : memref<?xf32, strided<[?], offset: ?>> to memref<?xf32, strided<[2], offset: ?>>
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

  // CHECK: %[[BUFFER:.*]] = scf.if %[[IS_FULL_TILE]] -> (memref<8xf32, strided<[1], offset: ?>>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1], offset: ?>>
  // CHECK:   scf.yield %[[SUBVIEW]] : memref<8xf32, strided<[1], offset: ?>>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [1] : memref<16xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<8xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0] [%[[SIZE]]] [1] : memref<8xf32> to memref<?xf32, strided<[1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
  // CHECK:   %[[CAST:.*]] = memref.cast %[[ALLOC_ELSE]] : memref<8xf32> to memref<8xf32, strided<[1], offset: ?>>
  // CHECK:   scf.yield %[[CAST]] : memref<8xf32, strided<[1], offset: ?>>
  // CHECK: }
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[BUFFER]] restrict : memref<8xf32, strided<[1], offset: ?>> to tensor<8xf32>
  %tile = xtile.extract %source[%tile_id][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @extract_5d(%[[SOURCE:.*]]: memref<1x2x1x32768x256xf32>, %[[D1:.*]]: index, %[[D3:.*]]: index, %[[D4:.*]]: index)
func.func @extract_5d(%source: memref<1x2x1x32768x256xf32>, %d1: index, %d3: index, %d4: index) -> tensor<1x1x1x1x16xf32> {
  // CHECK: %[[BUFFER:.*]] = scf.if {{.*}} -> (memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][0, %[[D1]], 0, %[[D3]], %[[D4]]] [1, 1, 1, 1, 16] [0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> to memref<1x1x1x1x16xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[CAST_IF:.*]] = memref.cast %[[SUBVIEW]] : memref<1x1x1x1x16xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>> to memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>>
  // CHECK:   scf.yield %[[CAST_IF]] : memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>>
  // CHECK: } else {
  // CHECK:   %[[INPUT_SUBVIEW:.*]] = memref.subview %{{.*}} : memref<1x2x1x32768x256xf32> to memref<?x?x?x?x?xf32, strided<[0, 8388608, 0, 256, 1], offset: ?>>
  // CHECK:   %[[ALLOC_ELSE:.*]] = memref.alloc() : memref<1x1x1x1x16xf32>
  // CHECK:   %[[ALLOC_SUBVIEW:.*]] = memref.subview %[[ALLOC_ELSE]][0, 0, 0, 0, 0] [1, %{{.*}}, 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1, 1] : memref<1x1x1x1x16xf32> to memref<1x?x1x?x?xf32, strided<[16, 16, 16, 16, 1]>>
  // CHECK:   memref.copy %[[INPUT_SUBVIEW]], %[[ALLOC_SUBVIEW]]
  // CHECK:   %[[CAST_ELSE:.*]] = memref.cast %[[ALLOC_ELSE]] : memref<1x1x1x1x16xf32> to memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>>
  // CHECK:   scf.yield %[[CAST_ELSE]] : memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>>
  // CHECK: }

  // CHECK: bufferization.to_tensor %[[BUFFER]] restrict : memref<1x1x1x1x16xf32, strided<[16, 16, 16, 16, 1], offset: ?>> to tensor<1x1x1x1x16xf32>
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0, %d1, %c0, %d3, %d4][1, 1, 1, 1, 16][0, 1, 0, 1, 1] : memref<1x2x1x32768x256xf32> -> tensor<1x1x1x1x16xf32>
  return %tile : tensor<1x1x1x1x16xf32>
}

// -----

// CHECK: @extract_static(%[[SOURCE:.*]]: memref<16xf32>)
func.func @extract_static(%source: memref<16xf32>) -> tensor<8xf32> {
  // CHECK-NOT: scf.if
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[SOURCE]][0] [8] [1] : memref<16xf32> to memref<8xf32, strided<[1]>>
  // CHECK: %[[TILE:.*]] = bufferization.to_tensor %[[SUBVIEW]] restrict : memref<8xf32, strided<[1]>> to tensor<8xf32>
  // CHECK: return %[[TILE]]
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0][8][1] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// Verify that small extracted tiles that theoretically require spill prevention 
// bypass the allocation and use a subview instead because they easily fit in 
// modern vector registers (e.g. <= 512 bytes).
//
// CHECK-LABEL: @extract_small_noalloc
func.func @extract_small_noalloc(%source: memref<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NOT: memref.alloc()
  // CHECK: %[[CAST:.*]] = memref.cast %arg0 : memref<16x8xf32> to memref<16x8xf32, strided<[8, 1]>>
  // CHECK: %[[RES:.*]] = bufferization.to_tensor %[[CAST]] restrict : memref<16x8xf32, strided<[8, 1]>> to tensor<16x8xf32>
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0, %c0][16, 8][1, 1] : memref<16x8xf32> -> tensor<16x8xf32>
  "dummy.usage"(%tile) : (tensor<16x8xf32>) -> ()
  return %tile : tensor<16x8xf32>
}

// -----

// Verify that large extracted tiles that require spill prevention 
// are correctly allocated to avoid register spilling in the generated assembly.
//
// CHECK-LABEL: @extract_large_broadcast_alloc
func.func @extract_large_broadcast_alloc(%source: memref<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16x32xf32>
  // CHECK: memref.copy %arg0, %[[ALLOC]] : memref<16x32xf32> to memref<16x32xf32>
  // CHECK: %[[CAST:.*]] = memref.cast %[[ALLOC]] : memref<16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
  // CHECK: %[[RES:.*]] = bufferization.to_tensor %[[CAST]] restrict : memref<16x32xf32, strided<[32, 1], offset: ?>> to tensor<16x32xf32>
  %c0 = arith.constant 0 : index
  %tile = xtile.extract %source[%c0, %c0][16, 32][1, 1] : memref<16x32xf32> -> tensor<16x32xf32>
  "dummy.broadcast"(%tile) : (tensor<16x32xf32>) -> ()
  return %tile : tensor<16x32xf32>
}

// -----
 
// Test: identity-layout extract feeding a simple element-wise op (arith.addf)
// with a small tile should yield the subview directly (no alloc in full-tile
// branch).
 
// CHECK-LABEL: @extract_identity_elementwise
func.func @extract_identity_elementwise(%source: memref<64xf32>) -> tensor<8xf32> {
  // 8 x f32 = 32 bytes, well below the 1536-byte register threshold, and
  // arith.addf is element-wise → no forced alloc needed.
 
  // CHECK-NOT: scf.if
  // CHECK-NOT: memref.alloc
 
  %c0 = arith.constant 0 : index
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[0] [8] [1] : memref<64xf32> to memref<8xf32, strided<[1]>>
  %tile = xtile.extract %source[%c0][8][1] : memref<64xf32> -> tensor<8xf32>
  // CHECK: %[[T1:.*]] = bufferization.to_tensor %[[SUBVIEW]] restrict : memref<8xf32, strided<[1]>> to tensor<8xf32>
  %tile2 = xtile.extract %source[%c0][8][1] : memref<64xf32> -> tensor<8xf32>
  // CHECK: %[[RESULT:.*]] = arith.addf %[[T1]], %[[T1]] : tensor<8xf32>
  %result = arith.addf %tile, %tile2 : tensor<8xf32>
  // CHECK: return %[[RESULT]] : tensor<8xf32>
  return %result : tensor<8xf32>
}
