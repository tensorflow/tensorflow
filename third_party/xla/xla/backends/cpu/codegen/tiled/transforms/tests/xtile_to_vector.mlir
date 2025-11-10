// RUN: fusion_compiler_opt %s --xtile-cpu-xtile-to-vector -cse -split-input-file | FileCheck %s

// CHECK-LABEL: @simple_insert_extract
// CHECK-SAME: (%[[INPUT:.*]]: memref<1024xf32>, %[[OUTPUT:.*]]: memref<1024xf32>, %[[TILE_ID:.*]]: index)
xtile.entry_func @simple_insert_extract(%input: memref<1024xf32>, %output: memref<1024xf32>, %tile_id: index) {
  // CHECK-DAG: %[[POISON:.*]] = ub.poison : f32
  // CHECK-DAG: %[[C_0:.*]] = arith.constant 0 : index
  // CHECK: %[[IN_SUBVIEW:.*]] = memref.subview %[[INPUT]][%[[TILE_ID]]] [1] [1]
  // CHECK-SAME: memref<1024xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: %[[MASK:.*]] = vector.create_mask
  // CHECK: %[[EXTRACT:.*]] = vector.transfer_read %[[IN_SUBVIEW]][%[[C_0]]], %[[POISON]], %[[MASK]]
  %tile = xtile.extract %input[%tile_id][1][1] : memref<1024xf32> -> tensor<1xf32>
  // CHECK: %[[OUT_SUBVIEW:.*]] = memref.subview %[[OUTPUT]][%[[TILE_ID]]] [1] [1]
  // CHECK-SAME: memref<1024xf32> to memref<1xf32, strided<[1], offset: ?>>
  // CHECK: vector.transfer_write %[[EXTRACT]], %[[OUT_SUBVIEW]][%[[C_0]]], %[[MASK]]
  xtile.insert %tile into %output[%tile_id][1][1] : tensor<1xf32> -> memref<1024xf32>
  xtile.return
}

// -----

// CHECK: @reduce_dimension(%[[INPUT:.*]]: memref<16x1024xf32>, %[[OUTPUT:.*]]: memref<16x1024xf32>, %[[TILE_ID:.*]]: index)
xtile.entry_func @reduce_dimension(%input: memref<16x1024xf32>, %output: memref<16x1024xf32>, %tile_id: index) {
  // CHECK: %[[C_0:.*]] = arith.constant 0 : index
  %offset = arith.constant 0 : index
  // CHECK: memref.subview %[[INPUT]][%[[C_0]], %[[TILE_ID]]] [10, 1] [1, 1]
  // CHECK-SAME: memref<16x1024xf32> to memref<10xf32, strided<[1024], offset: ?>>
  %tile = xtile.extract %input[%offset, %tile_id][10, 1][1, 1] : memref<16x1024xf32> -> tensor<10xf32>
  // CHECK: memref.subview %[[OUTPUT]][%[[C_0]], %[[TILE_ID]]] [10, 1] [1, 1]
  // CHECK-SAME: memref<16x1024xf32> to memref<10xf32, strided<[1024], offset: ?>>
  xtile.insert %tile into %output[%offset, %tile_id][10, 1][1, 1] : tensor<10xf32> -> memref<16x1024xf32>
  xtile.return
}

// -----

// CHECK: @extract_strided(%[[SOURCE:.*]]: memref<16xf32>, %[[TILE_ID:.*]]: index)
func.func @extract_strided(%source: memref<16xf32>, %tile_id: index) -> tensor<8xf32> {
  // CHECK: memref.subview %[[SOURCE]][%[[TILE_ID]]] [8] [2] :
  // CHECK-SAME: memref<16xf32> to memref<8xf32, strided<[2], offset: ?>>
  %tile = xtile.extract %source[%tile_id][8][2] : memref<16xf32> -> tensor<8xf32>
  return %tile : tensor<8xf32>
}

// -----

// CHECK: @insert_strided(
// CHECK-SAME: %[[SOURCE:.*]]: tensor<8xf32>,
// CHECK-SAME: %[[DESTINATION:.*]]: memref<16xf32>,
// CHECK-SAME: %[[TILE_ID:.*]]: index)
func.func @insert_strided(%source: tensor<8xf32>, %destination: memref<16xf32>, %tile_id: index) {
  // CHECK: memref.subview %[[DESTINATION]][%[[TILE_ID]]] [8] [2] :
  // CHECK-SAME: memref<16xf32> to memref<8xf32, strided<[2], offset: ?>>
  xtile.insert %source into %destination[%tile_id][8][2] : tensor<8xf32> -> memref<16xf32>
  return
}


