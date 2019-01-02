// RUN: mlir-opt %s -loop-tile -tile-size=32 | FileCheck %s

// CHECK-DAG: #map0 = (d0) -> (d0 + 32)
// CHECK-DAG: #map1 = (d0) -> (d0 + 32, 50)
// CHECK-DAG: [[IDENTITY:#map[0-9]+]] = (d0) -> (d0)
// CHECK-DAG: [[LB:#map[0-9]+]] = ()[s0] -> (0, s0)
// CHECK-DAG: [[UB:#map[0-9]+]] = ()[s0, s1] -> (s0, 4096 floordiv s1)
// CHECK-DAG: [[UB_INTRA_TILE:#map[0-9]+]] = (d0, d1, d2) -> (d2 + 32, s0, 4096 floordiv s1)

// CHECK-LABEL: func @loop_tiling()
// CHECK-NEXT:   for %i0 = 0 to 256 step 32 {
// CHECK-NEXT:     for %i1 = 0 to 512 step 32 {
// CHECK-NEXT:       for %i2 = 0 to 1024 step 32 {
// CHECK-NEXT:         for %i3 = [[IDENTITY]](%i0) to #map0(%i0) {
// CHECK-NEXT:           for %i4 = [[IDENTITY]](%i1) to #map0(%i1) {
// CHECK-NEXT:             for %i5 = [[IDENTITY]](%i2) to #map0(%i2) {
// CHECK-NEXT:               "foo"(%i3, %i4, %i5) : (index, index, index) -> ()
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   for %i6 = 0 to 50 step 32 {
// CHECK-NEXT:     for %i7 = [[IDENTITY]](%i6) to min #map1(%i6) {
// CHECK-NEXT:       "bar"(%i7, %i7) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: for %i8 = 0 to 21 step 32 {
// CHECK-NEXT:    for %i9 = [[IDENTITY]](%i8) to 21 {
// CHECK-NEXT:      "foobar"(%i9) : (index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func @loop_tiling() {
  for %i = 0 to 256 {
    for %j = 0 to 512 {
      for %k = 0 to 1024 {
        "foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }

  for %x = 0 to 50 {
    "bar"(%x, %x) : (index, index) -> ()
  }

  // Intra-tile loop won't need a min expression.
  for %y = 0 to 21 {
    "foobar"(%y) : (index) -> ()
  }

  return
}

#lb = ()[s0] -> (0, s0)
#ub = ()[s0, s1] -> (s0, 4096 floordiv s1)
// CHECK-LABEL: func @loop_max_min_bound(%arg0: memref<?xi32>, %arg1: index, %arg2: index) {
func @loop_max_min_bound(%A : memref<? x i32>, %L : index, %U : index) {
  %M = dim %A, 0 : memref<? x i32>
  for %iTT = max #lb()[%L] to min #ub()[%M, %U] {
      %out = affine_apply (d0) -> (d0) (%iTT)
  }
  return
// CHECK:       for %i0 = max [[LB]]()[%arg1] to min [[UB]]()[%0, %arg2] step 32 {
// CHECK-NEXT:    for %i1 = [[IDENTITY]](%i0) to min [[UB_INTRA_TILE]](%0, %arg2, %i0) {
// CHECK-NEXT:      %1 = affine_apply [[IDENTITY]](%i1)
// CHECK-NEXT:    }
// CHECK-NEXT:  }
}
