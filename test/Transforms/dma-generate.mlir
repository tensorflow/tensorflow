// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma -affine-data-copy-generate-fast-mem-space=2 -affine-data-copy-generate-skip-non-unit-stride-loops -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma -affine-data-copy-generate-fast-mem-capacity=16 -affine-data-copy-generate-fast-mem-space=2 | FileCheck %s --check-prefix FAST-MEM-16KB

// We run most test cases with -copy-skip-non-unit-stride-loops to allow testing
// DMA generation at inner levels easily - since the DMA generation would
// otherwise always generate DMAs at the outermost level (default for fast mem
// capacity is infinite). Using a specific capacity makes it harder to write
// a test case as one would have to calculate total footprints. With
// -copy-skip-non-unit-stride-loops, non-unit strides will always be skipped and
// its inner loops will be traversed till a unit stride loop is found (or the
// innermost block is reached).

// -----

// Index of the buffer for the second DMA is remapped.
// CHECK-DAG: [[MAP_PLUS_256:#map[0-9]+]] = (d0) -> (d0 + 256)
// CHECK-DAG: [[MAP0:#map[0-9]+]] = (d0) -> (d0)

// CHECK-LABEL: func @loop_nest_1d() {
func @loop_nest_1d() {
  %A = alloc() : memref<256 x f32>
  %B = alloc() : memref<512 x f32>
  %F = alloc() : memref<256 x f32, 2>
  // First DMA buffer.
  // CHECK:  %{{.*}} = alloc() : memref<256xf32>
  // CHECK:  %{{.*}} = alloc() : memref<256xf32, 2>
  // Tag for first DMA.
  // CHECK:  %{{.*}} = alloc() : memref<1xi32>
  // First DMA transfer.
  // CHECK:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32>, memref<256xf32, 2>, memref<1xi32>
  // CHECK:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
  // Second DMA buffer.
  // CHECK:  %{{.*}} = alloc() : memref<256xf32, 2>
  // Tag for second DMA.
  // CHECK:  %{{.*}} = alloc() : memref<1xi32>
  // Second DMA transfer.
  // CHECK:       affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<512xf32>, memref<256xf32, 2>, memref<1xi32>
  // CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
  // CHECK: affine.for %{{.*}} = 0 to 256 {
      // CHECK-NEXT: %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<256xf32, 2>
      // CHECK:      %{{.*}} = affine.apply [[MAP_PLUS_256]](%{{.*}})
      // Buffer for '%{{.*}}' in faster memref space is smaller size: 256xf32
      // Affine map for 'affine.load %{{.*}}' is composed: %{{.*}} + 256 - 256 = %{{.*}}.
      // CHECK-NEXT: %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<256xf32, 2>
      // Already in faster memory space.
      // CHECK:     %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<256xf32, 2>
  // CHECK-NEXT: }
  // CHECK-NEXT: dealloc %{{.*}} : memref<1xi32>
  // CHECK-NEXT: dealloc %{{.*}} : memref<256xf32, 2>
  // CHECK-NEXT: dealloc %{{.*}} : memref<1xi32>
  // CHECK-NEXT: dealloc %{{.*}} : memref<256xf32, 2>
  // CHECK-NEXT: return
  affine.for %i = 0 to 256 {
    affine.load %A[%i] : memref<256 x f32>
    %idx = affine.apply (d0) -> (d0 + 256)(%i)
    affine.load %B[%idx] : memref<512 x f32>
    affine.load %F[%i] : memref<256 x f32, 2>
  }
  return
}

// -----

// CHECK-LABEL: func @loop_nest_high_d
// CHECK:      %{{.*}} = constant 16384 : index
// CHECK-DAG:  [[BUFB:%[0-9]+]] = alloc() : memref<512x32xf32, 2>
// CHECK-DAG:  [[BUFA:%[0-9]+]] = alloc() : memref<512x32xf32, 2>
// CHECK-DAG:  [[BUFC:%[0-9]+]] = alloc() : memref<512x32xf32, 2>
// CHECK-DAG:  [[TAGB:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGA:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGC:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGC_W:%[0-9]+]] = alloc() : memref<1xi32>
// INCOMING DMA for B
// CHECK-DAG:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], [[BUFB]][%{{.*}}, %{{.*}}], [[TAGB]][%{{.*}}], %{{.*}} : memref<512x32xf32>, memref<512x32xf32, 2>, memref<1xi32>
// CHECK-DAG:  affine.dma_wait [[TAGB]][%{{.*}}], %{{.*}} : memref<1xi32>
// INCOMING DMA for A.
// CHECK-DAG:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], [[BUFA]][%{{.*}}, %{{.*}}], [[TAGA]][%{{.*}}], %{{.*}} : memref<512x32xf32>, memref<512x32xf32, 2>, memref<1xi32>
// CHECK-DAG:  affine.dma_wait [[TAGA]][%{{.*}}], %{{.*}} : memref<1xi32>
// INCOMING DMA for C.
// CHECK-DAG:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], [[BUFC]][%{{.*}}, %{{.*}}], [[TAGC]][%{{.*}}], %{{.*}} : memref<512x32xf32>, memref<512x32xf32, 2>, memref<1xi32>
// CHECK-DAG:  affine.dma_wait [[TAGC]][%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:          %{{.*}} = affine.apply #map{{[0-9]+}}(%{{.*}}, %{{.*}})
// CHECK-NEXT:          %{{.*}} = affine.load [[BUFB]][%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<512x32xf32, 2>
// CHECK-NEXT:          "foo"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:          %{{.*}} = affine.apply #map{{[0-9]+}}(%{{.*}}, %{{.*}})
// CHECK-NEXT:          %{{.*}} = affine.load [[BUFA]][%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<512x32xf32, 2>
// CHECK-NEXT:          "bar"(%{{.*}}) : (f32) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 16 {
// CHECK-NEXT:          %{{.*}} = "abc_compute"() : () -> f32
// CHECK-NEXT:          %{{.*}} = affine.apply #map{{[0-9]+}}(%{{.*}}, %{{.*}})
// CHECK-NEXT:          %{{.*}} = affine.load [[BUFC]][%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<512x32xf32, 2>
// CHECK-NEXT:          %{{.*}} = "addf32"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// CHECK-NEXT:          affine.store %{{.*}}, [[BUFC]][%{{.*}} * 16 + %{{.*}}, %{{.*}}] : memref<512x32xf32, 2>
// CHECK-NEXT:        }
// CHECK-NEXT:        "foobar"() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// OUTGOING DMA for C.
// CHECK-NEXT:  affine.dma_start [[BUFC]][%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], [[TAGC_W]][%{{.*}}], %{{.*}} : memref<512x32xf32, 2>, memref<512x32xf32>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait [[TAGC_W]][%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc [[TAGC_W]] : memref<1xi32>
// CHECK-NEXT:  dealloc [[TAGC]] : memref<1xi32>
// CHECK-NEXT:  dealloc [[BUFC]] : memref<512x32xf32, 2>
// CHECK-NEXT:  dealloc [[TAGA]] : memref<1xi32>
// CHECK-NEXT:  dealloc [[BUFA]] : memref<512x32xf32, 2>
// CHECK-NEXT:  dealloc [[TAGB]] : memref<1xi32>
// CHECK-NEXT:  dealloc [[BUFB]] : memref<512x32xf32, 2>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func @loop_nest_high_d(%A: memref<512 x 32 x f32>,
    %B: memref<512 x 32 x f32>, %C: memref<512 x 32 x f32>) {
  // DMAs will be performed at this level (jT is the first loop without a stride).
  // A and B are read, while C is both read and written. A total of three new buffers
  // are allocated and existing load's/store's are replaced by accesses to those buffers.
  affine.for %jT = 0 to 32 {
    affine.for %kT = 0 to 32 {
      affine.for %iT = 0 to 32 {
        affine.for %kk = 0 to 16 { // k intratile
          %k = affine.apply (d0, d1) -> (16*d0 + d1) (%kT, %kk)
          %v0 = affine.load %B[%k, %jT] : memref<512 x 32 x f32>
          "foo"(%v0) : (f32) -> ()
        }
        affine.for %ii = 0 to 16 { // i intratile.
          %i = affine.apply (d0, d1) -> (16*d0 + d1)(%iT, %ii)
          %v1 = affine.load %A[%i, %kT] : memref<512 x 32 x f32>
          "bar"(%v1) : (f32) -> ()
        }
        affine.for %ii_ = 0 to 16 { // i intratile.
          %v2 = "abc_compute"() : () -> f32
          %i_ = affine.apply (d0, d1) -> (16*d0 + d1)(%iT, %ii_)
          %v3 =  affine.load %C[%i_, %jT] : memref<512 x 32 x f32>
          %v4 = "addf32"(%v2, %v3) : (f32, f32) -> (f32)
          affine.store %v4, %C[%i_, %jT] : memref<512 x 32 x f32>
        }
        "foobar"() : () -> ()
      }
    }
  }
  return
}

// -----

// A loop nest with a modulo 2 access. A strided DMA is not needed here a 1x2
// region within a 256 x 8 memref.
//
// CHECK-LABEL: func @loop_nest_modulo() {
// CHECK:       %{{.*}} = alloc() : memref<256x8xf32>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 32 step 4 {
// CHECK-NEXT:      %{{.*}} = affine.apply #map{{[0-9]+}}(%{{.*}})
// CHECK-NEXT:      %{{.*}} = alloc() : memref<1x2xf32, 2>
// CHECK-NEXT:      %{{.*}} = alloc() : memref<1xi32>
// Composition of the affine map for '%{{.*}}' causes '%{{.*}}' to be added as a symbol.
// CHECK-NEXT:      affine.dma_start %{{.*}}[%{{.*}}, 0], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256x8xf32>, memref<1x2xf32, 2>, memref<1xi32>
// CHECK-NEXT:      affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 8 {
//                    ...
//                    ...
// CHECK:           }
// CHECK-NEXT:      dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:      dealloc %{{.*}} : memref<1x2xf32, 2>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func @loop_nest_modulo() {
  %A = alloc() : memref<256 x 8 x f32>
  affine.for %i = 0 to 32 step 4 {
    // DMAs will be performed at this level (%j is the first unit stride loop)
    affine.for %j = 0 to 8 {
      %idx = affine.apply (d0) -> (d0 mod 2) (%j)
      // A buffer of size 32 x 2 will be allocated (original buffer was 256 x 8).
      %v = affine.load %A[%i, %idx] : memref<256 x 8 x f32>
    }
  }
  return
}

// -----

// DMA on tiled loop nest. This also tests the case where the bounds are
// dependent on outer loop IVs.
// CHECK-LABEL: func @loop_nest_tiled() -> memref<256x1024xf32> {
func @loop_nest_tiled() -> memref<256x1024xf32> {
  %0 = alloc() : memref<256x1024xf32>
  affine.for %i0 = 0 to 256 step 32 {
    affine.for %i1 = 0 to 1024 step 32 {
// CHECK:      %{{.*}} = alloc() : memref<32x32xf32, 2>
// CHECK-NEXT: %{{.*}} = alloc() : memref<1xi32>
// Strided DMA here: 32 x 32 tile in a 256 x 1024 memref.
// CHECK-NEXT: affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}, %{{.*}} : memref<256x1024xf32>, memref<32x32xf32, 2>, memref<1xi32>
// CHECK-NEXT: affine.dma_wait
// CHECK-NEXT: affine.for %{{.*}} = #map
// CHECK-NEXT:   affine.for %{{.*}} = #map
      affine.for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 32)(%i0) {
        affine.for %i3 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 32)(%i1) {
          // CHECK: %{{.*}} = affine.load %{{.*}}[-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<32x32xf32, 2>
          %1 = affine.load %0[%i2, %i3] : memref<256x1024xf32>
        } // CHECK-NEXT: }
      }
    }
  }
  return %0 : memref<256x1024xf32>
}

// -----

// CHECK-LABEL: func @dma_constant_dim_access
func @dma_constant_dim_access(%A : memref<100x100xf32>) {
  %one = constant 1 : index
  %N = constant 100 : index
  // CHECK:      %{{.*}} = alloc() : memref<1x100xf32, 2>
  // CHECK-NEXT: %{{.*}} = alloc() : memref<1xi32>
  // No strided DMA needed here.
  // CHECK:      affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}  : memref<100x100xf32>, memref<1x100xf32, 2>,
  // CHECK-NEXT: affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to ()[s0] -> (s0) ()[%N] {
      // CHECK: %{{.*}} = affine.load %{{.*}}[0, %{{.*}}] : memref<1x100xf32, 2>
      affine.load %A[%one, %j] : memref<100 x 100 x f32>
    }
  }
  return
}

// -----

// CHECK-DAG: [[MAP_SYM_SHIFT:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d1 + s0 + s1)

// CHECK-LABEL: func @dma_with_symbolic_accesses
func @dma_with_symbolic_accesses(%A : memref<100x100xf32>, %M : index) {
  %N = constant 9 : index
  affine.for %i = 0 to 100 {
    affine.for %j = 0 to 100 {
      %idy = affine.apply (d0, d1) [s0, s1] -> (d1 + s0 + s1)(%i, %j)[%M, %N]
      affine.load %A[%i, %idy] : memref<100 x 100 x f32>
    }
  }
  return
// CHECK:       %{{.*}} = alloc() : memref<100x100xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[0, symbol(%{{.*}}) + 9], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}}
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 100 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 100 {
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_SYM_SHIFT]](%{{.*}}, %{{.*}})[%{{.*}}, %{{.*}}]
// CHECK-NEXT:      %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<100x100xf32, 2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK:       return
}

// -----

// CHECK-LABEL: func @dma_with_symbolic_loop_bounds
func @dma_with_symbolic_loop_bounds(%A : memref<100x100xf32>, %M : index, %N: index) {
  %K = constant 9 : index
// The buffer size can't be bound by a constant smaller than the original
// memref size; so the DMA buffer is the entire 100x100.
// CHECK:       %{{.*}} = alloc() : memref<100x100xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<100x100xf32>, memref<100x100xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
  affine.for %i = 0 to 100 {
    affine.for %j = %M to %N {
      %idy = affine.apply (d1) [s0] -> (d1 + s0)(%j)[%K]
      affine.load %A[%i, %idy] : memref<100 x 100 x f32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @dma_unknown_size
func @dma_unknown_size(%arg0: memref<?x?xf32>) {
  %M = dim %arg0, 0 : memref<? x ? x f32>
  %N = dim %arg0, 0 : memref<? x ? x f32>
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      // If this loop nest isn't tiled, the access requires a non-constant DMA
      // size -- not yet implemented.
      // CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
      affine.load %arg0[%i, %j] : memref<? x ? x f32>
      // expected-error@-6 {{copy generation failed for one or more memref's in this block}}
    }
  }
  return
}

// -----

// CHECK-LABEL: func @dma_memref_3d
func @dma_memref_3d(%arg0: memref<1024x1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %idx = affine.apply (d0) -> (d0 mod 128)(%i)
        %idy = affine.apply (d0) -> (d0 mod 128)(%j)
        %idz = affine.apply (d0) -> (d0 mod 128)(%k)
        // DMA with nested striding (or emulating with loop around strided DMA)
        // not yet implemented.
        // CHECK: %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<1024x1024x1024xf32>
        %v = affine.load %arg0[%idx, %idy, %idz] : memref<1024 x 1024 x 1024 x f32>
        // expected-error@-10 {{copy generation failed for one or more memref's in this block}}
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[MAP_PLUS_64:#map[0-9]+]] = (d0) -> (d0 + 64)
// CHECK-DAG: [[MAP_PLUS_128:#map[0-9]+]] = (d0) -> (d0 + 128)
// CHECK-DAG: [[MAP_PLUS_2:#map[0-9]+]] = (d0) -> (d0 + 2)
// CHECK-DAG: [[MAP_PLUS_192:#map[0-9]+]] = (d0) -> (d0 + 192)

// The first load accesses ([2,258), [128,384))
// The second load accesses ([64,320), [2,258))
// The first store writes to ([2,258), [192,448))
// The second store writes to ([128,320), [2,258))
// The union of all these regions is of size 318 x 446 and has its origin at (2,
// 2), i.e., the window ([2,320), [2,448)) in the original space.

// CHECK-LABEL: func @multi_load_store_union() {
func @multi_load_store_union() {
  %A = alloc() : memref<512 x 512 x f32>
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      %idx = affine.apply (d0) -> (d0 + 64)(%i)
      %idy = affine.apply (d0) -> (d0 + 128)(%j)
      %ishift = affine.apply (d0) -> (d0 + 2)(%i)
      %jshift = affine.apply (d0) -> (d0 + 2)(%j)

      %u = affine.load %A[%ishift, %idy] : memref<512 x 512 x f32>
      %v = affine.load %A[%idx, %jshift] : memref<512 x 512 x f32>

      %sidx = affine.apply (d0) -> (d0 + 128)(%i)
      %sidy = affine.apply (d0) -> (d0 + 192)(%j)

      affine.store %u, %A[%ishift, %sidy] : memref<512 x 512 x f32>
      affine.store %v, %A[%sidx, %jshift] : memref<512 x 512 x f32>
    }
  }
  return
}
// CHECK:       %{{.*}} = alloc() : memref<512x512xf32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<382x446xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}, %{{.*}} : memref<512x512xf32>, memref<382x446xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_64]](%{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_128]](%{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_2]](%{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_2]](%{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}} + 126] : memref<382x446xf32, 2>
// CHECK-NEXT:      %{{.*}} = affine.load %{{.*}}[%{{.*}} + 62, %{{.*}}] : memref<382x446xf32, 2>
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_128]](%{{.*}})
// CHECK-NEXT:      %{{.*}} = affine.apply [[MAP_PLUS_192]](%{{.*}})
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}} + 190] : memref<382x446xf32, 2>
// CHECK-NEXT:      affine.store %{{.*}}, %{{.*}}[%{{.*}} + 126, %{{.*}}] : memref<382x446xf32, 2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}}, %{{.*}} : memref<382x446xf32, 2>, memref<512x512xf32>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<382x446xf32, 2>
// CHECK-NEXT:  return
// CHECK-NEXT:}

// -----

// CHECK-LABEL: func @dma_loop_straightline_interspersed() {
func @dma_loop_straightline_interspersed() {
  %c0 = constant 0 : index
  %c255 = constant 255 : index
  %A = alloc() : memref<256 x f32>
  %v = affine.load %A[%c0] : memref<256 x f32>
  affine.for %i = 1 to 255 {
    affine.load %A[%i] : memref<256 x f32>
  }
  %l = affine.load %A[%c255] : memref<256 x f32>
  affine.store %l, %A[%c0] : memref<256 x f32>
  return
}
// There are three regions here - the 'load' preceding the loop, the loop
// itself, and the operations appearing after the loop.
// CHECK:       %{{.*}} = alloc() : memref<256xf32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32>, memref<1xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[0] : memref<1xf32, 2>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<254xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32>, memref<254xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 1 to 255 {
// CHECK-NEXT:    %{{.*}} = affine.load %{{.*}}[%{{.*}} - 1] : memref<254xf32, 2>
// CHECK-NEXT:  }
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<254xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<256xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32>, memref<256xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[255] : memref<256xf32, 2>
// CHECK-NEXT:  affine.store %{{.*}}, %{{.*}}[0] : memref<256xf32, 2>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32, 2>, memref<256xf32>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<1xi32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<256xf32, 2>
// CHECK-NEXT:  return

// -----

// CHECK-LABEL: func @dma_mixed_loop_blocks() {
func @dma_mixed_loop_blocks() {
  %c0 = constant 0 : index
  %A = alloc() : memref<256 x 256 x vector<8 x f32>>
  affine.for %i = 0 to 256 {
    %v = affine.load %A[%c0, %c0] : memref<256 x 256 x vector<8 x f32>>
    "foo"(%v) : (vector<8 x f32>) -> ()
    affine.for %j = 0 to 256 {
      %w = affine.load %A[%i, %j] : memref<256 x 256 x vector<8 x f32>>
      "bar"(%w) : (vector<8 x f32>) -> ()
    }
  }
  return
}
// CHECK-DAG:   [[MEM:%[0-9]+]] = alloc() : memref<256x256xvector<8xf32>>
// CHECK-DAG:   [[BUF:%[0-9]+]] = alloc() : memref<256x256xvector<8xf32>, 2>
// CHECK-DAG:   [[TAG:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK:       affine.dma_start [[MEM]][%{{.*}}, %{{.*}}], [[BUF]][%{{.*}}, %{{.*}}], [[TAG]][%{{.*}}], %{{.*}} : memref<256x256xvector<8xf32>>, memref<256x256xvector<8xf32>, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait [[TAG]][%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 256 {
// CHECK:         %{{.*}} = affine.load [[BUF]][0, 0] : memref<256x256xvector<8xf32>, 2>
// CHECK:         affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:      %{{.*}} = affine.load [[BUF]][%{{.*}}, %{{.*}}] : memref<256x256xvector<8xf32>, 2>

// -----

// CHECK-LABEL: func @relative_loop_bounds
func @relative_loop_bounds(%arg0: memref<1027xf32>) {
  affine.for %i0 = 0 to 1024 {
    affine.for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 4)(%i0) {
      %0 = constant 0.0 : f32
      affine.store %0, %arg0[%i2] : memref<1027xf32>
    }
  }
  return
}
// CHECK:      [[BUF:%[0-9]+]] = alloc() : memref<1027xf32, 2>
// CHECK-NEXT: [[MEM:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-NEXT: affine.for %{{.*}} = 0 to 1024 {
// CHECK-NEXT:    affine.for %{{.*}} = {{#map[0-9]+}}(%{{.*}}) to {{#map[0-9]+}}(%{{.*}}) {
// CHECK-NEXT:      %{{.*}} = constant 0.000000e+00 : f32
// CHECK-NEXT:      affine.store %{{.*}}, [[BUF]][%{{.*}}] : memref<1027xf32, 2>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.dma_start [[BUF]][%{{.*}}], %{{.*}}[%{{.*}}], [[MEM]][%{{.*}}], %{{.*}}  : memref<1027xf32, 2>, memref<1027xf32>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait [[MEM]][%{{.*}}], %{{.*}} : memref<1xi32>

// -----

// CHECK-DAG: [[MAP_READ_OFFSET:#map[0-9]+]] = (d0) -> (d0 + 100)
// CHECK-DAG: [[MAP_WRITE_OFFSET:#map[0-9]+]] = (d0) -> (d0 + 25)

func @test_read_write_region_union() {
  %0 = alloc() : memref<256xf32>
  affine.for %i0 = 0 to 10 {
    // memref dims:  [0, 256)
    // read region:  [100, 110)
    // write region: [25, 35)
    // union region: [25, 110)
    %a0 = affine.apply (d0) -> (d0 + 100)(%i0)
    %a1 = affine.apply (d0) -> (d0 + 25)(%i0)
    %1 = affine.load %0[%a0] : memref<256xf32>
    affine.store %1, %0[%a1] : memref<256xf32>
  }
  return
}

// CHECK:       %{{.*}} = alloc() : memref<256xf32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<85xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<256xf32>, memref<85xf32, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
// CHECK-NEXT:    %{{.*}} = affine.apply [[MAP_READ_OFFSET]](%{{.*}})
// CHECK-NEXT:    %{{.*}} = affine.apply [[MAP_WRITE_OFFSET]](%{{.*}})
// CHECK-NEXT:    %{{.*}} = affine.load %{{.*}}[%{{.*}} + 75] : memref<85xf32, 2>
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<85xf32, 2>
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<85xf32, 2>, memref<256xf32>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>

// -----

// This should create a buffer of size 2 affine.for %arg2.

#map_lb = (d0) -> (d0)
#map_ub = (d0) -> (d0 + 3)
#map_acc = (d0) -> (d0 floordiv 8)
// CHECK-LABEL: func @test_analysis_util
func @test_analysis_util(%arg0: memref<4x4x16x1xf32>, %arg1: memref<144x9xf32>, %arg2: memref<2xf32>) -> (memref<144x9xf32>, memref<2xf32>) {
  %c0 = constant 0 : index
  %0 = alloc() : memref<64x1xf32>
  %1 = alloc() : memref<144x4xf32>
  %2 =  constant 0.0 : f32
  affine.for %i8 = 0 to 9 step 3 {
    affine.for %i9 = #map_lb(%i8) to #map_ub(%i8) {
      affine.for %i17 = 0 to 64 {
        %23 = affine.apply #map_acc(%i9)
        %25 = affine.load %arg2[%23] : memref<2xf32>
        %26 = affine.apply #map_lb(%i17)
        %27 = affine.load %0[%26, %c0] : memref<64x1xf32>
        affine.store %27, %arg2[%23] : memref<2xf32>
      }
    }
  }
  return %arg1, %arg2 : memref<144x9xf32>, memref<2xf32>
}
// CHECK:       affine.for %{{.*}} = 0 to 9 step 3 {
// CHECK:         [[BUF:%[0-9]+]] = alloc() : memref<2xf32, 2>
// CHECK:         affine.dma_start %{{.*}}[%{{.*}} floordiv 8], [[BUF]]
// CHECK:         affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>
// CHECK:         affine.for %{{.*}} =

// ----

#map3 = (d0) -> (d0)
#map12 = (d0) -> (d0 + 3)
#map14 = (d0, d1) -> ((d0 + d1 * 72) floordiv 2304 + ((((d0 + d1 * 72) mod 2304) mod 1152) mod 9) floordiv 3)
#map15 = (d0, d1) -> ((d0 + d1 * 72) mod 2304 - (((d0 + d1 * 72) mod 2304) floordiv 1152) * 1151 - ((((d0 + d1 * 72) mod 2304) mod 1152) floordiv 9) * 9 - (((((d0 + d1 * 72) mod 2304) mod 1152) mod 9) floordiv 3) * 3)
#map16 = (d0, d1) -> (((((d0 + d1 * 72) mod 2304) mod 1152) floordiv 9) floordiv 8)
// Test for test case in b/128303048 #4.
func @test_memref_bounds(%arg0: memref<4x4x16x1xvector<8x128xf32>>, %arg1: memref<144x9xvector<8x128xf32>>, %arg2: memref<2xvector<8x128xf32>>) -> (memref<144x9xvector<8x128xf32>>, memref<2xvector<8x128xf32>>) {
  %c0 = constant 0 : index
  affine.for %i8 = 0 to 9 step 3 {
    affine.for %i9 = #map3(%i8) to #map12(%i8) {
      affine.for %i10 = 0 to 64 {
        %10 = affine.apply #map14(%i9, %i10)
        %11 = affine.apply #map15(%i9, %i10)
        %12 = affine.apply #map16(%i9, %i10)
        %13 = affine.load %arg0[%10, %11, %12, %c0] : memref<4x4x16x1xvector<8x128xf32>>
      }
    }
  }
  return %arg1, %arg2 : memref<144x9xvector<8x128xf32>>, memref<2xvector<8x128xf32>>
}

// CHECK:       %{{.*}} = alloc() : memref<4x4x16x1xvector<8x128xf32>, 2>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<1xi32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], %{{.*}} : memref<4x4x16x1xvector<8x128xf32>>, memref<4x4x16x1xvector<8x128xf32>, 2>, memref<1xi32>
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xi32>

// -----

// Since the fast memory size is 4 KB, DMA generation will happen right under
// %i0.

// FAST-MEM-16KB-LABEL: func @load_store_same_memref
func @load_store_same_memref(%arg0: memref<256x1024xf32>) {
  // FAST-MEM-16KB:  affine.for %{{.*}} = 0 to 256 step 4
  affine.for %i0 = 0 to 256 step 4 {
    // FAST-MEM-16KB: [[BUF:%[0-9]+]] = alloc() : memref<4x1024xf32, 2>
    // FAST-MEM-16KB:    affine.dma_start %{{.*}}
    // FAST-MEM-16KB-NEXT: affine.dma_wait
    // FAST-MEM-16KB:  affine.for %{{.*}}
    affine.for %i1 = 0 to 1024 step 4 {
      // FAST-MEM-16KB:  affine.for %{{.*}}
      affine.for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 4)(%i0) {
        // FAST-MEM-16KB:  affine.for %{{.*}}
        affine.for %i3 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 4)(%i1) {
          %3 = affine.load %arg0[%i2, %i3] : memref<256x1024xf32>
          %4 = mulf %3, %3 : f32
          affine.store %4, %arg0[%i2, %i3] : memref<256x1024xf32>
        } // FAST-MEM-16KB: }
      } // FAST-MEM-16KB: }
    } // FAST-MEM-16KB: }
    // FAST-MEM-16KB:    affine.dma_start [[BUF]]
    // FAST-MEM-16KB-NEXT: affine.dma_wait
  }
  return
}

// -----

// This a 3-d loop nest tiled by 4 x 4 x 4. Under %i, %j, %k, the size of a
// tile of arg0, arg1, and arg2 accessed is 4 KB (each), i.e., 12 KB in total.
// With fast mem capacity set to 16 KB, the DMAs if placed under %k will fit.
// However, the region of arg2 accessed is invariant w.r.t the %k loop unlike
// %arg0 and %arg1. So, its DMA can be hoisted one level up and placed under
// %j, while the DMAs for arg0 and arg1 appear right under the %k loop.

#map0 = (d0) -> (d0)
#map1 = (d0) -> (d0 + 4)
// FAST-MEM-16KB-LABEL: func @simple_matmul
func @simple_matmul(%arg0: memref<8x8xvector<64xf32>>, %arg1: memref<8x8xvector<64xf32>>, %arg2: memref<8x8xvector<64xf32>>) -> memref<8x8xvector<64xf32>> {
  affine.for %i = 0 to 8 step 4 {
    affine.for %j = 0 to 8 step 4 {
      affine.for %k = 0 to 8 step 4 {
        affine.for %ii = #map0(%i) to #map1(%i) {
          affine.for %jj = #map0(%j) to #map1(%j) {
            affine.for %kk = #map0(%k) to #map1(%k) {
              %5 = affine.load %arg0[%ii, %kk] : memref<8x8xvector<64xf32>>
              %6 = affine.load %arg1[%kk, %jj] : memref<8x8xvector<64xf32>>
              %7 = affine.load %arg2[%ii, %jj] : memref<8x8xvector<64xf32>>
              %8 = mulf %5, %6 : vector<64xf32>
              %9 = addf %7, %8 : vector<64xf32>
              affine.store %9, %arg2[%ii, %jj] : memref<8x8xvector<64xf32>>
            }
          }
        }
      }
    }
  }
  return %arg2 : memref<8x8xvector<64xf32>>
}
// FAST-MEM-16KB: affine.for %{{.*}} = 0 to 8 step 4 {
// FAST-MEM-16KB:   affine.for %{{.*}} = 0 to 8 step 4 {
// FAST-MEM-16KB:     affine.dma_start %{{.*}}
// FAST-MEM-16KB:     affine.dma_wait
// FAST-MEM-16KB:     affine.for %{{.*}} = 0 to 8 step 4 {
// FAST-MEM-16KB:       affine.dma_start %{{.*}}
// FAST-MEM-16KB:       affine.dma_wait
// FAST-MEM-16KB:       affine.dma_start %{{.*}}
// FAST-MEM-16KB:       affine.dma_wait
// FAST-MEM-16KB:       affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to #map{{[0-9]+}}(%{{.*}}) {
// FAST-MEM-16KB-NEXT:    affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to #map{{[0-9]+}}(%{{.*}}) {
// FAST-MEM-16KB-NEXT:      affine.for %{{.*}} = #map{{[0-9]+}}(%{{.*}}) to #map{{[0-9]+}}(%{{.*}}) {
// FAST-MEM-16KB:           }
// FAST-MEM-16KB:         }
// FAST-MEM-16KB:       }
// FAST-MEM-16KB:     }
// FAST-MEM-16KB:     affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}
// FAST-MEM-16KB:     affine.dma_wait
