// RUN: mlir-opt %s -split-input-file | FileCheck %s

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (d0, d1)

// Test with loop IVs.
func @test0(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  %1 = alloc() : memref<100x100xf32, (d0, d1) -> (d0, d1), 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.dma_start %0[%i0, %i1], %1[%i0, %i1], %2[%c0], %c64
        : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
      affine.dma_wait %2[%c0], %c64 : memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK: affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (d0, d1)

// Test with loop IVs and optional stride arguments.
func @test1(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  %1 = alloc() : memref<100x100xf32, (d0, d1) -> (d0, d1), 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  %c128 = constant 128 : index
  %c256 = constant 256 : index
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.dma_start %0[%i0, %i1], %1[%i0, %i1], %2[%c0], %c64, %c128, %c256
        : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
      affine.dma_wait %2[%c0], %c64 : memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}, %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK: affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2) -> (d0, d1 + d2 + 5)
// CHECK: [[MAP1:#map[0-9]+]] = (d0, d1, d2) -> (d0 + d1, d2)

// Test with loop IVs and symbols (without symbol keyword).
func @test2(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  %1 = alloc() : memref<100x100xf32, (d0, d1) -> (d0, d1), 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.dma_start %0[%i0 + %arg0, %i1], %1[%i0, %i1 + %arg1 + 5],
                       %2[%c0], %c64
        : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
      affine.dma_wait %2[%c0], %c64 : memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[%{{.*}} + %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}} + %{{.*}} + 5], %{{.*}}[%{{.*}}], %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK: affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1)[s0] -> (d0, d1 + s0 + 7)
// CHECK: [[MAP1:#map[0-9]+]] = (d0, d1)[s0] -> (d0 + s0, d1)
// CHECK: [[MAP1:#map[0-9]+]] = (d0, d1) -> (d0 + d1 + 11)

// Test with loop IVs and symbols (with symbol keyword).
func @test3(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  %1 = alloc() : memref<100x100xf32, (d0, d1) -> (d0, d1), 2>
  %2 = alloc() : memref<1xi32>
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.dma_start %0[%i0 + symbol(%arg0), %i1],
                       %1[%i0, %i1 + symbol(%arg1) + 7],
                       %2[%i0 + %i1 + 11], %c64
        : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
      affine.dma_wait %2[%c0], %c64 : memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[%{{.*}} + symbol(%{{.*}}), %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}} + symbol(%{{.*}}) + 7], %{{.*}}[%{{.*}} + %{{.*}} + 11], %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK: affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1)[s0] -> (d0, (d1 + s0) mod 9 + 7)
// CHECK: [[MAP1:#map[0-9]+]] = (d0, d1)[s0] -> ((d0 + s0) floordiv 3, d1)
// CHECK: [[MAP2:#map[0-9]+]] = (d0, d1) -> (d0 + d1 + 11)

// Test with loop IVs, symbols and constants in nested affine expressions.
func @test4(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  %1 = alloc() : memref<100x100xf32, 2>
  %2 = alloc() : memref<1xi32>
  %c64 = constant 64 : index
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.dma_start %0[(%i0 + symbol(%arg0)) floordiv 3, %i1],
                       %1[%i0, (%i1 + symbol(%arg1)) mod 9 + 7],
                       %2[%i0 + %i1 + 11], %c64
        : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
      affine.dma_wait %2[%i0 + %i1 + 11], %c64 : memref<1xi32>
// CHECK: affine.dma_start %{{.*}}[(%{{.*}} + symbol(%{{.*}})) floordiv 3, %{{.*}}], %{{.*}}[%{{.*}}, (%{{.*}} + symbol(%{{.*}})) mod 9 + 7], %{{.*}}[%{{.*}} + %{{.*}} + 11], %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK: affine.dma_wait %{{.*}}[%{{.*}} + %{{.*}} + 11], %{{.*}} : memref<1xi32>
    }
  }
  return
}
