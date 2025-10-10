// RUN: emitters_opt %s --xtile-cpu-xtile-to-vector -split-input-file | FileCheck %s

// CHECK-LABEL: @simple_insert_extract
// CHECK-SAME: (%[[INPUT:.*]]: memref<1024xf32>, %[[OUTPUT:.*]]: memref<1024xf32>, %[[TILE_ID:.*]]: index)
xtile.entry_func @simple_insert_extract(%input: memref<1024xf32>, %output: memref<1024xf32>, %tile_id: index) {
  // CHECK-DAG: %[[POISON:.*]] = ub.poison : f32
  // CHECK: %[[EXTRACT:.*]] = vector.transfer_read %[[INPUT]][%[[TILE_ID]]], %[[POISON]] : memref<1024xf32>, vector<1xf32>
  %tile = xtile.extract %input[%tile_id][1][1] : memref<1024xf32> -> tensor<1xf32>
  // CHECK: vector.transfer_write %[[EXTRACT]], %[[OUTPUT]][%[[TILE_ID]]] : vector<1xf32>, memref<1024xf32>
  xtile.insert %tile into %output[%tile_id][1][1] : tensor<1xf32> -> memref<1024xf32>
  xtile.return
}


// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: @reduce_dimension(%[[INPUT:.*]]: memref<16x1024xf32>, %[[OUTPUT:.*]]: memref<16x1024xf32>, %[[TILE_ID:.*]]: index)
xtile.entry_func @reduce_dimension(%input: memref<16x1024xf32>, %output: memref<16x1024xf32>, %tile_id: index) {
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  %offset = arith.constant 0 : index
  // CHECK: %[[POISON:.*]] = ub.poison : f32
  // CHECK: %[[EXTRACT:.*]] = vector.transfer_read %[[INPUT]][%[[OFFSET]], %[[TILE_ID]]], %[[POISON]] {in_bounds = [true], permutation_map = #[[MAP]]} : memref<16x1024xf32>, vector<10xf32>
  // CHECK: vector.transfer_write %[[EXTRACT]], %[[OUTPUT]][%[[OFFSET]], %[[TILE_ID]]] {in_bounds = [true], permutation_map = #[[MAP]]} : vector<10xf32>, memref<16x1024xf32>
  %tile = xtile.extract %input[%offset, %tile_id][10, 1][1, 1] : memref<16x1024xf32> -> tensor<10xf32>
  xtile.insert %tile into %output[%offset, %tile_id][10, 1][1, 1] : tensor<10xf32> -> memref<16x1024xf32>
  xtile.return
}

// -----
