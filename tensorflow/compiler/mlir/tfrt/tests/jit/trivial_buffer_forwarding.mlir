// RUN: tf-tfrt-opt %s -split-input-file                                       \
// RUN:                -tf-jitrt-linalg-trivial-buffer-forwarding              \
// RUN:   | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @reuse_input_buffer
func.func @reuse_input_buffer(%arg0: index) -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[IN2:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg0) : memref<?x10xf32>
  %2 = memref.alloc(%arg0) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)
  linalg.fill ins(%cst_1 : f32) outs(%1 : memref<?x10xf32>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]], %[[IN2]] :  memref<?x10xf32>, memref<?x10xf32>)
  // CHECK-SAME: outs(%[[IN]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map0, #map0],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0, %1 : memref<?x10xf32>, memref<?x10xf32>) outs(%2 : memref<?x10xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %3 = arith.addf %lhs, %rhs : f32
      linalg.yield %3 : f32
    }
  // CHECK: memref.dealloc %[[OUT]]
  memref.dealloc %0 : memref<?x10xf32>
  memref.dealloc %1 : memref<?x10xf32>

  // CHECK: return %[[IN]]
  func.return %2: memref<?x10xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// Do not forward input operand if output buffer has user before linalg.generic.
// CHECK-LABEL: @user_before_generic
func.func @user_before_generic(%arg0: index) -> memref<?x?xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  // CHECK: %[[CASTED_OUT:.*]] = memref.cast
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg0) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)
  %2 = memref.cast %1 : memref<?x10xf32> to memref<?x?xf32>

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32>)
  // CHECK-SAME: outs(%[[OUT]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map0],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = math.sqrt %in : f32
      linalg.yield %3 : f32
    }
  // CHECK: memref.dealloc %[[IN]]
  memref.dealloc %0 : memref<?x10xf32>

  // CHECK: return %[[CASTED_OUT]]
  func.return %2: memref<?x?xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// Do not forward input operand with non-identity indexing map.
// CHECK-LABEL: @non_identity_input_indexing_map
func.func @non_identity_input_indexing_map(%arg0: index) -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg0) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32>)
  // CHECK-SAME: outs(%[[OUT]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map1, #map0],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.sqrt %in : f32
      linalg.yield %2 : f32
    }
  // CHECK: memref.dealloc %[[IN]]
  memref.dealloc %0 : memref<?x10xf32>

  // CHECK: return %[[OUT]]
  func.return %1: memref<?x10xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1 * 4)>

// Do not forward input operand with non-contiguous memory layout.
// CHECK-LABEL: @non_contiguous_input_memref
func.func @non_contiguous_input_memref(%arg0: index) -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32, #map1>
  %1 = memref.alloc(%arg0) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32, #map1>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32, #map
  // CHECK-SAME: outs(%[[OUT]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map0],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32, #map1>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.sqrt %in : f32
      linalg.yield %2 : f32
    }
  // CHECK: memref.dealloc %[[IN]]
  memref.dealloc %0 : memref<?x10xf32, #map1>

  // CHECK: return %[[OUT]]
  func.return %1: memref<?x10xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// Forward same size with non-identiy maps.
// CHECK-LABEL: @non_identity_same_size
func.func @non_identity_same_size(%arg0: index) -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg0) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32>)
  // CHECK-SAME: outs(%[[IN]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map1],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.sqrt %in : f32
      linalg.yield %2 : f32
    }
  // CHECK: memref.dealloc %[[OUT]]
  memref.dealloc %0 : memref<?x10xf32>

  // CHECK: return %[[IN]]
  func.return %1: memref<?x10xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// Do not forward different size with non-identiy maps.
// CHECK-LABEL: @non_identity_different_size
func.func @non_identity_different_size(%arg0: index, %arg1: index)
    -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg1) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32>)
  // CHECK-SAME: outs(%[[OUT]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map1],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.sqrt %in : f32
      linalg.yield %2 : f32
    }
  // CHECK: memref.dealloc %[[IN]]
  memref.dealloc %0 : memref<?x10xf32>

  // CHECK: return %[[OUT]]
  func.return %1: memref<?x10xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// Forward different size with identiy maps.
// CHECK-LABEL: @identity_different_size
func.func @identity_different_size(%arg0: index, %arg1: index) -> memref<?x10xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK: %[[IN:.*]] = memref.alloc
  // CHECK: %[[OUT:.*]] = memref.alloc
  %0 = memref.alloc(%arg0) : memref<?x10xf32>
  %1 = memref.alloc(%arg1) : memref<?x10xf32>
  linalg.fill ins(%cst_0 : f32) outs(%0 : memref<?x10xf32>)

  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[IN]] :  memref<?x10xf32>)
  // CHECK-SAME: outs(%[[IN]] :  memref<?x10xf32>)
  linalg.generic { indexing_maps = [#map0, #map0],
                   iterator_types = ["parallel", "parallel"] }
  ins(%0 : memref<?x10xf32>) outs(%1 : memref<?x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.sqrt %in : f32
      linalg.yield %2 : f32
    }
  // CHECK: memref.dealloc %[[OUT]]
  memref.dealloc %0 : memref<?x10xf32>

  // CHECK: return %[[IN]]
  func.return %1: memref<?x10xf32>
}
