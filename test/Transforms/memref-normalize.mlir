// RUN: mlir-opt -simplify-affine-structures %s | FileCheck %s

// CHECK-LABEL: func @permute()
func @permute() {
  %A = alloc() : memref<64x256xf32, (d0, d1) -> (d1, d0)>
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 256 {
      affine.load %A[%i, %j] : memref<64x256xf32, (d0, d1) -> (d1, d0)>
    }
  }
  dealloc %A : memref<64x256xf32, (d0, d1) -> (d1, d0)>
  return
}
// The old memref alloc should disappear.
// CHECK-NOT:  memref<64x256xf32>
// CHECK:      [[MEM:%[0-9]+]] = alloc() : memref<256x64xf32>
// CHECK-NEXT: affine.for %[[I:arg[0-9]+]] = 0 to 64 {
// CHECK-NEXT:   affine.for %[[J:arg[0-9]+]] = 0 to 256 {
// CHECK-NEXT:     affine.load [[MEM]][%[[J]], %[[I]]] : memref<256x64xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: dealloc [[MEM]]
// CHECK-NEXT: return

// CHECK-LABEL: func @shift()
func @shift() {
  // CHECK-NOT:  memref<64xf32, (d0) -> (d0 + 1)>
  %A = alloc() : memref<64xf32, (d0) -> (d0 + 1)>
  affine.for %i = 0 to 64 {
    affine.load %A[%i] : memref<64xf32, (d0) -> (d0 + 1)>
    // CHECK: %{{.*}} = affine.load %{{.*}}[%arg{{.*}} + 1] : memref<65xf32>
  }
  return
}

// CHECK-LABEL: func @high_dim_permute()
func @high_dim_permute() {
  // CHECK-NOT: memref<64x128x256xf32,
  %A = alloc() : memref<64x128x256xf32, (d0, d1, d2) -> (d2, d0, d1)>
  // CHECK: %[[I:arg[0-9]+]]
  affine.for %i = 0 to 64 {
    // CHECK: %[[J:arg[0-9]+]]
    affine.for %j = 0 to 128 {
      // CHECK: %[[K:arg[0-9]+]]
      affine.for %k = 0 to 256 {
        affine.load %A[%i, %j, %k] : memref<64x128x256xf32, (d0, d1, d2) -> (d2, d0, d1)>
        // CHECK: %{{.*}} = affine.load %{{.*}}[%[[K]], %[[I]], %[[J]]] : memref<256x64x128xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: func @invalid_map
func @invalid_map() {
  %A = alloc() : memref<64x128xf32, (d0, d1) -> (d0, -d1 - 10)>
  // CHECK: %{{.*}} = alloc() : memref<64x128xf32,
  return
}

// A tiled layout.
// CHECK-LABEL: func @data_tiling()
func @data_tiling() {
  %A = alloc() : memref<64x512xf32, (d0, d1) -> (d0 floordiv 8, d1 floordiv 16, d0 mod 8, d1 mod 16)>
  // CHECK: %{{.*}} = alloc() : memref<8x32x8x16xf32>
  return
}


// Memref escapes; no normalization.
// CHECK-LABEL: func @escaping() -> memref<64xf32, #map{{[0-9]+}}>
func @escaping() ->  memref<64xf32, (d0) -> (d0 + 2)> {
  // CHECK: %{{.*}} = alloc() : memref<64xf32, #map{{[0-9]+}}>
  %A = alloc() : memref<64xf32, (d0) -> (d0 + 2)>
  return %A : memref<64xf32, (d0) -> (d0 + 2)>
}
