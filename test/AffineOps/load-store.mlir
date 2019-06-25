// RUN: mlir-opt %s -split-input-file | FileCheck %s

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (d0, d1)

// Test with just loop IVs.
func @test0(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0, %i1] : memref<100x100xf32>
// CHECK: %1 = affine.load %0[%i0, %i1] : memref<100x100xf32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1) -> (d0 + 3, d1 + 7)

// Test with loop IVs and constants.
func @test1(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
      affine.store %1, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
// CHECK: %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
// CHECK: affine.store %1, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2, d3) -> (d0 + d1, d2 + d3)

// Test with loop IVs and function args without 'symbol' keyword (should
// be parsed as dim identifiers).
func @test2(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
      affine.store %1, %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
// CHECK: %1 = affine.load %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
// CHECK: affine.store %1, %0[%i0 + %arg0, %i1 + %arg1] : memref<100x100xf32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)

// Test with loop IVs and function args with 'symbol' keyword (should
// be parsed as symbol identifiers).
func @test3(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
        : memref<100x100xf32>
      affine.store %1, %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)]
        : memref<100x100xf32>
// CHECK: %1 = affine.load %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)] : memref<100x100xf32>
// CHECK: affine.store %1, %0[%i0 + symbol(%arg0), %i1 + symbol(%arg1)] : memref<100x100xf32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1)[s0, s1] -> ((d0 + s0) floordiv 3 + 11, (d1 + s1) mod 4 + 7)

// Test with loop IVs, symbols and constants in nested affine expressions.
func @test4(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<100x100xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %1 = affine.load %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
      affine.store %1, %0[(%i0 + symbol(%arg0)) floordiv 3 + 11,
                          (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
// CHECK: %1 = affine.load %0[(%i0 + symbol(%arg0)) floordiv 3 + 11, (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
// CHECK: affine.store %1, %0[(%i0 + symbol(%arg0)) floordiv 3 + 11, (%i1 + symbol(%arg1)) mod 4 + 7] : memref<100x100xf32>
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2) -> (d0, d1, d2)

// Test with swizzled loop IVs.
func @test5(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2, %i0, %i1] : memref<10x10x10xf32>
        affine.store %1, %0[%i2, %i0, %i1] : memref<10x10x10xf32>
// CHECK: %1 = affine.load %0[%i2, %i0, %i1] : memref<10x10x10xf32>
// CHECK: affine.store %1, %0[%i2, %i0, %i1] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2, d3, d4) -> (d0 + d1, d2 + d3, d3 + d1 + d4)

// Test with swizzled loop IVs, duplicate args, and function args used as dims.
// Dim identifiers are assigned in parse order:
// d0 = %i2, d1 = %arg0, d2 = %i0, d3 = %i1, d4 = %arg1
func @test6(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1]
          : memref<10x10x10xf32>
        affine.store %1, %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1]
          : memref<10x10x10xf32>
// CHECK: %1 = affine.load %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1] : memref<10x10x10xf32>
// CHECK: affine.store %1, %0[%i2 + %arg0, %i0 + %i1, %i1 + %arg0 + %arg1] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0, d1, d2)[s0, s1] -> (d0 + s0, d1 + d2, d2 + s0 + s1)

// Test with swizzled loop IVs, duplicate args, and function args used as syms.
// Dim and symbol identifiers are assigned in parse order:
// d0 = %i2, d1 = %i0, d2 = %i1
// s0 = %arg0, s1 = %arg1
func @test6(%arg0 : index, %arg1 : index) {
  %0 = alloc() : memref<10x10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.for %i2 = 0 to 10 {
        %1 = affine.load %0[%i2 + symbol(%arg0),
                            %i0 + %i1,
                            %i1 + symbol(%arg0) + symbol(%arg1)]
                              : memref<10x10x10xf32>
        affine.store %1, %0[%i2 + symbol(%arg0),
                             %i0 + %i1,
                             %i1 + symbol(%arg0) + symbol(%arg1)]
                              : memref<10x10x10xf32>
// CHECK: %1 = affine.load %0[%i2 + symbol(%arg0), %i0 + %i1, %i1 + symbol(%arg0) + symbol(%arg1)] : memref<10x10x10xf32>
// CHECK: affine.store %1, %0[%i2 + symbol(%arg0), %i0 + %i1, %i1 + symbol(%arg0) + symbol(%arg1)] : memref<10x10x10xf32>
      }
    }
  }
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]+]] = (d0) -> (d0 + 1)

// Test with operands without special SSA name.
func @test7() {
  %0 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %1 = affine.apply (d1) -> (d1 + 1)(%i0)
    %2 = affine.load %0[%1] : memref<10xf32>
    affine.store %2, %0[%1] : memref<10xf32>
// CHECK: affine.load %0[%1] : memref<10xf32>
// CHECK: affine.store %2, %0[%1] : memref<10xf32>
  }
  return
}