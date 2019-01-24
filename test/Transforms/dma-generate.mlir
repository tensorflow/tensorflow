// RUN: mlir-opt %s -split-input-file -dma-generate -verify | FileCheck %s

// Index of the buffer for the second DMA is remapped.
// CHECK-DAG: [[MAP:#map[0-9]+]] = (d0) -> (d0 - 256)
// CHECK-DAG: #map{{[0-9]+}} = (d0, d1) -> (d0 * 16 + d1)
// CHECK-DAG: #map{{[0-9]+}} = (d0, d1) -> (d0, d1)
// CHECK-DAG: [[MAP_INDEX_DIFF:#map[0-9]+]] = (d0, d1, d2, d3) -> (d2 - d0, d3 - d1)
// CHECK-DAG: [[MAP_MINUS_ONE:#map[0-9]+]] = (d0, d1) -> (d0 - 1, d1)
// CHECK-DAG: [[MAP_ORIG_ACCESS:#map[0-9]+]] = (d0, d1)[s0, s1] -> (d0, d1 + s0 + s1)
// CHECK-DAG: [[MAP_SUB_OFFSET:#map[0-9]+]] = (d0, d1, d2) -> (d1, d2 - (d0 + 9))

// CHECK-LABEL: func @loop_nest_1d() {
func @loop_nest_1d() {
  %A = alloc() : memref<256 x f32>
  %B = alloc() : memref<512 x f32>
  %F = alloc() : memref<256 x f32, 1>
  // First DMA buffer.
  // CHECK:  %0 = alloc() : memref<256xf32>
  // CHECK:  %3 = alloc() : memref<256xf32, 1>
  // Tag for first DMA.
  // CHECK:  %4 = alloc() : memref<1xi32>
  // First DMA transfer.
  // CHECK:  dma_start %0[%c0], %3[%c0], %c256_1, %4[%c0] : memref<256xf32>, memref<256xf32, 1>, memref<1xi32>
  // CHECK:  dma_wait %4[%c0], %c256_1 : memref<1xi32>
  // Second DMA buffer.
  // CHECK:  %5 = alloc() : memref<256xf32, 1>
  // Tag for second DMA.
  // CHECK:  %6 = alloc() : memref<1xi32>
  // Second DMA transfer.
  // CHECK:       dma_start %1[%c256], %5[%c0], %c256_0, %6[%c0] : memref<512xf32>, memref<256xf32, 1>, memref<1xi32>
  // CHECK-NEXT:  dma_wait %6[%c0], %c256_0 : memref<1xi32>
  // CHECK: for %i0 = 0 to 256 {
      // CHECK:      %7 = affine_apply #map{{[0-9]+}}(%i0)
      // CHECK-NEXT: %8 = load %3[%7] : memref<256xf32, 1>
      // CHECK:      %9 = affine_apply #map{{[0-9]+}}(%i0)
      // CHECK:      %10 = affine_apply [[MAP]](%9)
      // CHECK-NEXT: %11 = load %5[%10] : memref<256xf32, 1>
      // Already in faster memory space.
      // CHECK:     %12 = load %2[%i0] : memref<256xf32, 1>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  for %i = 0 to 256 {
    load %A[%i] : memref<256 x f32>
    %idx = affine_apply (d0) -> (d0 + 256)(%i)
    load %B[%idx] : memref<512 x f32>
    load %F[%i] : memref<256 x f32, 1>
  }
  return
}

// CHECK-LABEL: func @loop_nest_high_d
// CHECK:      %c16384 = constant 16384 : index
// CHECK-DAG:  [[BUFB:%[0-9]+]] = alloc() : memref<512x32xf32, 1>
// CHECK-DAG:  [[BUFA:%[0-9]+]] = alloc() : memref<512x32xf32, 1>
// CHECK-DAG:  [[BUFC:%[0-9]+]] = alloc() : memref<512x32xf32, 1>
// CHECK-DAG:  [[TAGB:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGA:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGC:%[0-9]+]] = alloc() : memref<1xi32>
// CHECK-DAG:  [[TAGC_W:%[0-9]+]] = alloc() : memref<1xi32>
// INCOMING DMA for B
// CHECK-DAG:  dma_start %arg1[%c0, %c0], [[BUFB]][%c0, %c0], %c16384_2, [[TAGB]][%c0] : memref<512x32xf32>, memref<512x32xf32, 1>, memref<1xi32>
// CHECK-DAG:  dma_wait [[TAGB]][%c0], %c16384_2 : memref<1xi32>
// INCOMING DMA for A.
// CHECK-DAG:  dma_start %arg0[%c0, %c0], [[BUFA]][%c0, %c0], %c16384_1, [[TAGA]][%c0] : memref<512x32xf32>, memref<512x32xf32, 1>, memref<1xi32>
// CHECK-DAG:  dma_wait [[TAGA]][%c0], %c16384_1 : memref<1xi32>
// INCOMING DMA for C.
// CHECK-DAG:  dma_start %arg2[%c0, %c0], [[BUFC]][%c0, %c0], %c16384_0, [[TAGC]][%c0] : memref<512x32xf32>, memref<512x32xf32, 1>, memref<1xi32>
// CHECK-DAG:  dma_wait [[TAGC]][%c0], %c16384_0 : memref<1xi32>
// CHECK-NEXT:  for %i0 = 0 to 32 {
// CHECK-NEXT:    for %i1 = 0 to 32 {
// CHECK-NEXT:      for %i2 = 0 to 32 {
// CHECK-NEXT:        for %i3 = 0 to 16 {
// CHECK-NEXT:          %7 = affine_apply #map{{[0-9]+}}(%i1, %i3)
// CHECK-NEXT:          %8 = affine_apply #map{{[0-9]+}}(%7, %i0)
// CHECK-NEXT:          %9 = load [[BUFB]][%8#0, %8#1] : memref<512x32xf32, 1>
// CHECK-NEXT:          "foo"(%9) : (f32) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        for %i4 = 0 to 16 {
// CHECK-NEXT:          %10 = affine_apply #map{{[0-9]+}}(%i2, %i4)
// CHECK-NEXT:          %11 = affine_apply #map{{[0-9]+}}(%10, %i1)
// CHECK-NEXT:          %12 = load [[BUFA]][%11#0, %11#1] : memref<512x32xf32, 1>
// CHECK-NEXT:          "bar"(%12) : (f32) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:        for %i5 = 0 to 16 {
// CHECK-NEXT:          %13 = "abc_compute"() : () -> f32
// CHECK-NEXT:          %14 = affine_apply #map{{[0-9]+}}(%i2, %i5)
// CHECK-NEXT:          %15 = affine_apply #map{{[0-9]+}}(%14, %i0)
// CHECK-NEXT:          %16 = load [[BUFC]][%15#0, %15#1] : memref<512x32xf32, 1>
// CHECK-NEXT:          %17 = "addf32"(%13, %16) : (f32, f32) -> f32
// CHECK-NEXT:          %18 = affine_apply #map{{[0-9]+}}(%14, %i0)
// CHECK-NEXT:          store %17, [[BUFC]][%18#0, %18#1] : memref<512x32xf32, 1>
// CHECK-NEXT:        }
// CHECK-NEXT:        "foobar"() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// OUTGOING DMA for C.
// CHECK-NEXT:  dma_start [[BUFC]][%c0, %c0], %arg2[%c0, %c0], %c16384, [[TAGC_W]][%c0] : memref<512x32xf32, 1>, memref<512x32xf32>, memref<1xi32>
// CHECK-NEXT:  dma_wait [[TAGC_W]][%c0], %c16384 : memref<1xi32>
// CHECK-NEXT:  return
// CHECK-NEXT:}
func @loop_nest_high_d(%A: memref<512 x 32 x f32>,
    %B: memref<512 x 32 x f32>, %C: memref<512 x 32 x f32>) {
  // DMAs will be performed at this level (jT is the first loop without a stride).
  // A and B are read, while C is both read and written. A total of three new buffers
  // are allocated and existing load's/store's are replaced by accesses to those buffers.
  for %jT = 0 to 32 {
    for %kT = 0 to 32 {
      for %iT = 0 to 32 {
        for %kk = 0 to 16 { // k intratile
          %k = affine_apply (d0, d1) -> (16*d0 + d1) (%kT, %kk)
          %v0 = load %B[%k, %jT] : memref<512 x 32 x f32>
          "foo"(%v0) : (f32) -> ()
        }
        for %ii = 0 to 16 { // i intratile.
          %i = affine_apply (d0, d1) -> (16*d0 + d1)(%iT, %ii)
          %v1 = load %A[%i, %kT] : memref<512 x 32 x f32>
          "bar"(%v1) : (f32) -> ()
        }
        for %ii_ = 0 to 16 { // i intratile.
          %v2 = "abc_compute"() : () -> f32
          %i_ = affine_apply (d0, d1) -> (16*d0 + d1)(%iT, %ii_)
          %v3 =  load %C[%i_, %jT] : memref<512 x 32 x f32>
          %v4 = "addf32"(%v2, %v3) : (f32, f32) -> (f32)
          store %v4, %C[%i_, %jT] : memref<512 x 32 x f32>
        }
        "foobar"() : () -> ()
      }
    }
  }
  return
}

// A loop nest with a modulo 2 access. A strided DMA is not needed here a 1x2
// region within a 256 x 8 memref.
//
// CHECK-LABEL: func @loop_nest_modulo() {
// CHECK:       %0 = alloc() : memref<256x8xf32>
// CHECK-NEXT:    for %i0 = 0 to 32 step 4 {
// CHECK-NEXT:      %1 = affine_apply #map{{[0-9]+}}(%i0)
// CHECK-NEXT:      %2 = alloc() : memref<1x2xf32, 1>
// CHECK-NEXT:      %3 = alloc() : memref<1xi32>
// CHECK-NEXT:      dma_start %0[%1, %c0], %2[%c0, %c0], %c2, %3[%c0] : memref<256x8xf32>, memref<1x2xf32, 1>, memref<1xi32>
// CHECK-NEXT:      dma_wait %3[%c0], %c2 : memref<1xi32>
// CHECK-NEXT:      for %i1 = 0 to 8 {
//                    ...
//                    ...
// CHECK:           }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func @loop_nest_modulo() {
  %A = alloc() : memref<256 x 8 x f32>
  for %i = 0 to 32 step 4 {
    // DMAs will be performed at this level (%j is the first unit stride loop)
    for %j = 0 to 8 {
      %idx = affine_apply (d0) -> (d0 mod 2) (%j)
      // A buffer of size 32 x 2 will be allocated (original buffer was 256 x 8).
      %v = load %A[%i, %idx] : memref<256 x 8 x f32>
    }
  }
  return
}

// DMA on tiled loop nest. This also tests the case where the bounds are
// dependent on outer loop IVs.
// CHECK-LABEL: func @loop_nest_tiled() -> memref<256x1024xf32> {
func @loop_nest_tiled() -> memref<256x1024xf32> {
  %0 = alloc() : memref<256x1024xf32>
  for %i0 = 0 to 256 step 32 {
    for %i1 = 0 to 1024 step 32 {
// CHECK:      %3 = alloc() : memref<32x32xf32, 1>
// CHECK-NEXT: %4 = alloc() : memref<1xi32>
// Strided DMA here: 32 x 32 tile in a 256 x 1024 memref.
// CHECK-NEXT: dma_start %0[%1, %2], %3[%c0, %c0], %c1024, %4[%c0], %c1024_0, %c32 : memref<256x1024xf32>, memref<32x32xf32, 1>, memref<1xi32>
// CHECK-NEXT: dma_wait
// CHECK-NEXT: for %i2 = #map
// CHECK-NEXT:   for %i3 = #map
      for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 32)(%i0) {
        for %i3 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 32)(%i1) {
          // CHECK:      %5 = affine_apply [[MAP_INDEX_DIFF]](%i0, %i1, %i2, %i3)
          // CHECK-NEXT: %6 = load %3[%5#0, %5#1] : memref<32x32xf32, 1>
          %1 = load %0[%i2, %i3] : memref<256x1024xf32>
        } // CHECK-NEXT: }
      }
    }
  }
  // CHECK: return %0 : memref<256x1024xf32>
  return %0 : memref<256x1024xf32>
}

// CHECK-LABEL: func @dma_constant_dim_access
func @dma_constant_dim_access(%A : memref<100x100xf32>) {
  %one = constant 1 : index
  %N = constant 100 : index
  // CHECK:      %0 = alloc() : memref<1x100xf32, 1>
  // CHECK-NEXT: %1 = alloc() : memref<1xi32>
  // No strided DMA needed here.
  // CHECK:      dma_start %arg0[%c1, %c0], %0[%c0, %c0], %c100, %1[%c0] : memref<100x100xf32>, memref<1x100xf32, 1>,
  // CHECK-NEXT: dma_wait %1[%c0], %c100 : memref<1xi32>
  for %i = 0 to 100 {
    for %j = 0 to ()[s0] -> (s0) ()[%N] {
      // CHECK:      %2 = affine_apply [[MAP_MINUS_ONE]](%c1_0, %i1)
      // CHECK-NEXT: %3 = load %0[%2#0, %2#1] : memref<1x100xf32, 1>
      load %A[%one, %j] : memref<100 x 100 x f32>
    }
  }
  return
}

// CHECK-LABEL: func @dma_with_symbolic_accesses
func @dma_with_symbolic_accesses(%A : memref<100x100xf32>, %M : index) {
  %N = constant 9 : index
  for %i = 0 to 100 {
    for %j = 0 to 100 {
      %idx = affine_apply (d0, d1) [s0, s1] -> (d0, d1 + s0 + s1)(%i, %j)[%M, %N]
      load %A[%idx#0, %idx#1] : memref<100 x 100 x f32>
    }
  }
  return
// CHECK:       %1 = alloc() : memref<100x100xf32, 1>
// CHECK-NEXT:  %2 = alloc() : memref<1xi32>
// CHECK-NEXT:  dma_start %arg0[%c0, %0], %1[%c0, %c0], %c10000, %2[%c0]
// CHECK-NEXT:  dma_wait %2[%c0], %c10000
// CHECK-NEXT:  for %i0 = 0 to 100 {
// CHECK-NEXT:    for %i1 = 0 to 100 {
// CHECK-NEXT:      %3 = affine_apply [[MAP_ORIG_ACCESS]](%i0, %i1)[%arg1, %c9]
// CHECK-NEXT:      %4 = affine_apply [[MAP_SUB_OFFSET]](%arg1, %3#0, %3#1)
// CHECK-NEXT:      %5 = load %1[%4#0, %4#1] : memref<100x100xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// CHECK-LABEL: func @dma_with_symbolic_loop_bounds
func @dma_with_symbolic_loop_bounds(%A : memref<100x100xf32>, %M : index, %N: index) {
  %K = constant 9 : index
// The buffer size can't be bound by a constant smaller than the original
// memref size; so the DMA buffer is the entire 100x100.
// CHECK:       %0 = alloc() : memref<100x100xf32, 1>
// CHECK-NEXT:  %1 = alloc() : memref<1xi32>
// CHECK-NEXT:  dma_start %arg0[%c0, %c0], %0[%c0, %c0], %c10000, %1[%c0] : memref<100x100xf32>, memref<100x100xf32, 1>, memref<1xi32>
// CHECK-NEXT:  dma_wait %1[%c0], %c10000 : memref<1xi32>
  for %i = 0 to 100 {
    for %j = %M to %N {
      %idx = affine_apply (d0, d1) [s0] -> (d0, d1 + s0)(%i, %j)[%K]
      load %A[%idx#0, %idx#1] : memref<100 x 100 x f32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @dma_unknown_size
func @dma_unknown_size(%arg0: memref<?x?xf32>) {
  %M = dim %arg0, 0 : memref<? x ? x f32>
  %N = dim %arg0, 0 : memref<? x ? x f32>
  for %i = 0 to %M {
    for %j = 0 to %N {
      // If this loop nest isn't tiled, the access requires a non-constant DMA
      // size -- not yet implemented.
      // CHECK: %2 = load %arg0[%i0, %i1] : memref<?x?xf32>
      load %arg0[%i, %j] : memref<? x ? x f32>
      // expected-error@-6 {{DMA generation failed for one or more memref's}}
    }
  }
  return
}

// -----

// CHECK-LABEL: func @dma_memref_3d
func @dma_memref_3d(%arg0: memref<1024x1024x1024xf32>) {
  for %i = 0 to 1024 {
    for %j = 0 to 1024 {
      for %k = 0 to 1024 {
        %idx = affine_apply (d0, d1, d2) -> (d0 mod 128, d1 mod 128, d2 mod 128)(%i, %j, %k)
        // DMA with nested striding (or emulating with loop around strided DMA)
        // not yet implemented.
        // CHECK: %3 = load %arg0[%2#0, %2#1, %2#2] : memref<1024x1024x1024xf32>
        %v = load %arg0[%idx#0, %idx#1, %idx#2] : memref<1024 x 1024 x 1024 x f32>
        // expected-error@-8 {{DMA generation failed for one or more memref's}}
      }
    }
  }
  return
}
