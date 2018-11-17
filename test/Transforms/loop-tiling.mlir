// RUN: mlir-opt %s -loop-tile -tile-size=32 | FileCheck %s

// CHECK: #map0 = (d0) -> (d0 + 32)
// CHECK: #map1 = (d0) -> (d0 + 32, 50)
// CHECK-LABEL: mlfunc @loop_tiling()
// CHECK-NEXT:   for %i0 = 0 to 256 step 32 {
// CHECK-NEXT:     for %i1 = 0 to 512 step 32 {
// CHECK-NEXT:       for %i2 = 0 to 1024 step 32 {
// CHECK-NEXT:         for %i3 = (d0) -> (d0)(%i0) to #map0(%i0) {
// CHECK-NEXT:           for %i4 = (d0) -> (d0)(%i1) to #map0(%i1) {
// CHECK-NEXT:             for %i5 = (d0) -> (d0)(%i2) to #map0(%i2) {
// CHECK-NEXT:               "foo"(%i3, %i4, %i5) : (index, index, index) -> ()
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   for %i6 = 0 to 50 step 32 {
// CHECK-NEXT:     for %i7 = (d0) -> (d0)(%i6) to min #map1(%i6) {
// CHECK-NEXT:       "bar"(%i7, %i7) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: for %i8 = 0 to 21 step 32 {
// CHECK-NEXT:    for %i9 = (d0) -> (d0)(%i8) to 21 {
// CHECK-NEXT:      "foobar"(%i9) : (index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
mlfunc @loop_tiling() {
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
