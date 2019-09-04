// RUN: mlir-opt %s -split-input-file -affine-pipeline-data-transfer | FileCheck %s

// -----

// CHECK-DAG: [[MOD_2:#map[0-9]+]] = (d0) -> (d0 mod 2)
// CHECK-DAG: [[MAP_MINUS_1:#map[0-9]+]] = (d0) -> (d0 - 1)

// CHECK-LABEL: func @loop_nest_dma() {
func @loop_nest_dma() {

  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>

  %tag = alloc() : memref<1 x f32>

  %zero = constant 0 : index
  %num_elts = constant 32 : index

  affine.for %i = 0 to 8 {
    affine.dma_start %A[%i], %Ah[%i], %tag[%zero], %num_elts : memref<256 x f32>, memref<32 x f32, 1>, memref<1 x f32>
    affine.dma_wait %tag[%zero], %num_elts : memref<1 x f32>
    %v = affine.load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    %r = "compute"(%v) : (f32) -> (f32)
    affine.store %r, %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    affine.for %j = 0 to 32 {
      "do_more_compute"(%i, %j) : (index, index) -> ()
    }
  }
  dealloc %tag : memref<1 x f32>
  dealloc %Ah : memref<32 x f32, (d0) -> (d0), 1>
  return
}
// CHECK:       %{{.*}} = alloc() : memref<256xf32>
// CHECK:       %{{.*}} = alloc() : memref<2x32xf32, 1>
// CHECK-NEXT:  %{{.*}} = alloc() : memref<2x1xf32>
// CHECK-NEXT:  affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}} mod 2, %{{.*}}], %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 1 to 8 {
// CHECK-NEXT:    affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}} mod 2, %{{.*}}], %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:    %{{.*}} = affine.apply [[MAP_MINUS_1]](%{{.*}})
// CHECK-NEXT:    %{{.*}} = affine.apply [[MOD_2]](%{{.*}})
// CHECK-NEXT:    %{{.*}} = affine.apply [[MOD_2]](%{{.*}})
// CHECK-NEXT:    affine.dma_wait %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<2x1xf32>
// CHECK-NEXT:    %{{.*}} = affine.load %{{.*}}[%{{.*}} mod 2, %{{.*}}] : memref<2x32xf32, 1>
// CHECK-NEXT:    %{{.*}} = "compute"(%{{.*}}) : (f32) -> f32
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 2, %{{.*}}] : memref<2x32xf32, 1>
// CHECK-NEXT:    affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:      "do_more_compute"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %{{.*}} = affine.apply [[MAP_MINUS_1]](%{{.*}})
// CHECK-NEXT:  %{{.*}} = affine.apply [[MOD_2]](%{{.*}})
// CHECK-NEXT:  %{{.*}} = affine.apply [[MOD_2]](%{{.*}})
// CHECK-NEXT:  affine.dma_wait %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<2x1xf32>
// CHECK-NEXT:  %{{.*}} = affine.load %{{.*}}[%{{.*}} mod 2, %{{.*}}] : memref<2x32xf32, 1>
// CHECK-NEXT:  %{{.*}} = "compute"(%{{.*}}) : (f32) -> f32
// CHECK-NEXT:  affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 2, %{{.*}}] : memref<2x32xf32, 1>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to 32 {
// CHECK-NEXT:    "do_more_compute"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  dealloc %{{.*}} : memref<2x1xf32>
// CHECK-NEXT:  dealloc %{{.*}} : memref<2x32xf32, 1>
// CHECK-NEXT:  return
// CHECK-NEXT:}

// -----

// CHECK-DAG: [[FLOOR_MOD_2:#map[0-9]+]] = (d0) -> ((d0 floordiv 4) mod 2)
// CHECK-DAG: [[REMAP_SHIFT_MINUS_4:#map[0-9]+]] = (d0) -> (d0 - 4)

// CHECK-LABEL: @loop_step
func @loop_step(%arg0: memref<512xf32>,
                  %arg1: memref<512xf32>) {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  affine.for %i0 = 0 to 512 step 4 {
    %1 = alloc() : memref<4xf32, 1>
    %2 = alloc() : memref<1xi32>
    affine.dma_start %arg0[%i0], %1[%c0], %2[%c0], %c4,
              : memref<512xf32>, memref<4xf32, 1>, memref<1xi32>
    affine.dma_wait %2[%c0], %c4 : memref<1xi32>
    "compute"(%i0) : (index) -> ()
    dealloc %2 : memref<1xi32>
    dealloc %1 : memref<4xf32, 1>
  }
  return
}
// CHECK:        [[BUF:%[0-9]+]] = alloc() : memref<2x4xf32, 1>
// CHECK:        [[TAG:%[0-9]+]] = alloc() : memref<2x1xi32>
// CHECK-NEXT:   affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[(%{{.*}} floordiv 4) mod 2, 0], [[TAG]][(%{{.*}} floordiv 4) mod 2, 0], %{{.*}} : memref<512xf32>, memref<2x4xf32, 1>, memref<2x1xi32>
// CHECK-NEXT:   affine.for %{{.*}} = 4 to 512 step 4 {
// CHECK-NEXT:     affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[(%{{.*}} floordiv 4) mod 2, 0], [[TAG]][(%{{.*}} floordiv 4) mod 2, 0], %{{.*}} : memref<512xf32>, memref<2x4xf32, 1>, memref<2x1xi32>
// CHECK-NEXT:     %{{.*}} = affine.apply [[REMAP_SHIFT_MINUS_4]](%{{.*}})
// CHECK-NEXT:     %{{.*}} = affine.apply [[FLOOR_MOD_2]](%{{.*}})
// CHECK:          affine.dma_wait [[TAG]][(%{{.*}} floordiv 4) mod 2, 0], %{{.*}} : memref<2x1xi32>
// CHECK-NEXT:     "compute"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   [[SHIFTED:%[0-9]+]] = affine.apply [[REMAP_SHIFT_MINUS_4]](%{{.*}})
// CHECK-NEXT:   %{{.*}} = affine.apply [[FLOOR_MOD_2]]([[SHIFTED]])
// CHECK:        affine.dma_wait [[TAG]][(%{{.*}} floordiv 4) mod 2, 0], %{{.*}} : memref<2x1xi32>
// CHECK-NEXT:   "compute"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:   dealloc [[TAG]] : memref<2x1xi32>
// CHECK-NEXT:   dealloc [[BUF]] : memref<2x4xf32, 1>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

#map1 = (d0, d1) -> ((d0 * 2048 + d1 * 256) floordiv 32)
#map2 = (d0) -> ((d0 * 2048) floordiv 32)
// CHECK-LABEL: func @loop_dma_nested(%{{.*}}: memref<512x32xvector<8xf32>
func @loop_dma_nested(%arg0: memref<512x32xvector<8xf32>>, %arg1: memref<512x32xvector<8xf32>>, %arg2: memref<512x32xvector<8xf32>>) {
  %num_elts = constant 256 : index
  %c0 = constant 0 : index
  %0 = alloc() : memref<64x4xvector<8xf32>, 2>
  %1 = alloc() : memref<64x4xvector<8xf32>, 2>
  %2 = alloc() : memref<64x4xvector<8xf32>, 2>
  %3 = alloc() : memref<2xi32>
  %4 = alloc() : memref<2xi32>
  %5 = alloc() : memref<2xi32>
  // Prologue for DMA overlap on arg2.
  // CHECK-DAG: [[BUF_ARG2:%[0-9]+]] = alloc() : memref<2x64x4xvector<8xf32>, 2>
  // CHECK-DAG: [[TAG_ARG2:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK: affine.dma_start %{{.*}}[
  // CHECK: affine.for %{{.*}} = 1 to 8 {
  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    affine.dma_start %arg2[%6, %c0], %2[%c0, %c0], %5[%c0], %num_elts : memref<512x32xvector<8xf32>>, memref<64x4xvector<8xf32>, 2>, memref<2xi32>
    affine.dma_wait %5[%c0], %num_elts : memref<2xi32>
    // Steady state for DMA overlap on arg2
    // CHECK: affine.dma_start %{{.*}}[
    // CHECK: affine.dma_wait [[TAG_ARG2]]
    // Prologue for DMA overlap on arg0, arg1 nested within i0
    // CHECK: [[BUF_ARG0:%[0-9]+]] = alloc() : memref<2x64x4xvector<8xf32>, 2>
    // CHECK: [[BUF_ARG1:%[0-9]+]] = alloc() : memref<2x64x4xvector<8xf32>, 2>
    // CHECK: [[TAG_ARG0:%[0-9]+]] = alloc() : memref<2x2xi32>
    // CHECK: [[TAG_ARG1:%[0-9]+]] = alloc() : memref<2x2xi32>
    // CHECK: affine.dma_start %{{.*}}[
    // CHECK: affine.dma_start %{{.*}}[
    // CHECK-NEXT affine.for %{{.*}} = 1 to 8 {
    affine.for %i1 = 0 to 8 {
      %7 = affine.apply #map1(%i0, %i1)
      %8 = affine.apply #map2(%i1)
      affine.dma_start %arg0[%7, %c0], %0[%c0, %c0], %3[%c0], %num_elts : memref<512x32xvector<8xf32>>, memref<64x4xvector<8xf32>, 2>, memref<2xi32>
      affine.dma_start %arg1[%8, %c0], %1[%c0, %c0], %4[%c0], %num_elts : memref<512x32xvector<8xf32>>, memref<64x4xvector<8xf32>, 2>, memref<2xi32>
      affine.dma_wait %3[%c0], %num_elts : memref<2xi32>
      affine.dma_wait %4[%c0], %num_elts : memref<2xi32>
      // Steady state for DMA overlap on arg0, arg1
      // CHECK: affine.dma_start %{{.*}}[
      // CHECK: affine.dma_start %{{.*}}[
      // CHECK: affine.dma_wait [[TAG_ARG0]]
      // CHECK: affine.dma_wait [[TAG_ARG1]]
      // CHECK-NEXT: affine.for %{{.*}} = 0 to 4 {
      affine.for %i2 = 0 to 4 {
        "foo"() : () -> ()
      }
    }
    // epilogue for arg0, arg1
    // CHECK: affine.dma_wait [[TAG_ARG0]]
    // CHECK: affine.dma_wait [[TAG_ARG1]]
    // CHECK-DAG:    dealloc [[TAG_ARG1]] : memref<2x2xi32>
    // CHECK-DAG:    dealloc [[TAG_ARG0]] : memref<2x2xi32>
    // CHECK-DAG:    dealloc [[BUF_ARG1]] : memref<2x64x4xvector<8xf32>, 2>
    // CHECK-DAG:    dealloc [[BUF_ARG0]] : memref<2x64x4xvector<8xf32>, 2>
  // epilogue for DMA overlap on %arg2
  // CHECK:  affine.dma_wait [[TAG_ARG2]]
  // Within the epilogue for arg2's DMA, we have the DMAs on %arg1, %arg2 nested.
  // CHECK: [[BUF_ARG0_NESTED:%[0-9]+]] = alloc() : memref<2x64x4xvector<8xf32>, 2>
  // CHECK: [[BUF_ARG1_NESTED:%[0-9]+]] = alloc() : memref<2x64x4xvector<8xf32>, 2>
  // CHECK: [[TAG_ARG0_NESTED:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK: [[TAG_ARG1_NESTED:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK:  affine.dma_start %{{.*}}[
  // CHECK:  affine.dma_start %{{.*}}[
  // CHECK:  affine.for %{{.*}} = 1 to 8 {
  // CHECK:    affine.dma_start %{{.*}}[
  // CHECK:    affine.dma_start %{{.*}}[
  // CHECK:    affine.dma_wait [[TAG_ARG0_NESTED]]
  // CHECK:    affine.dma_wait [[TAG_ARG1_NESTED]]
  // CHECK:    affine.for %{{.*}} = 0 to 4 {
  // CHECK:      "foo"() : () -> ()
  // CHECK:  affine.dma_wait [[TAG_ARG0_NESTED]]
  // CHECK:  affine.dma_wait [[TAG_ARG1_NESTED]]
  // CHECK:  affine.for %{{.*}} = 0 to 4 {
  }
  dealloc %5 : memref<2xi32>
  dealloc %4 : memref<2xi32>
  dealloc %3 : memref<2xi32>
  dealloc %2 : memref<64x4xvector<8xf32>, 2>
  dealloc %1 : memref<64x4xvector<8xf32>, 2>
  dealloc %0 : memref<64x4xvector<8xf32>, 2>
  return
// CHECK: }
// CHECK-DAG:  dealloc [[TAG_ARG1_NESTED]] : memref<2x2xi32>
// CHECK-DAG:  dealloc [[TAG_ARG0_NESTED]] : memref<2x2xi32>
// CHECK-DAG:  dealloc [[BUF_ARG1_NESTED]] : memref<2x64x4xvector<8xf32>, 2>
// CHECK-DAG:  dealloc [[BUF_ARG0_NESTED]] : memref<2x64x4xvector<8xf32>, 2>
// CHECK-DAG:  dealloc [[TAG_ARG2]] : memref<2x2xi32>
// CHECK-DAG:  dealloc [[BUF_ARG2]] : memref<2x64x4xvector<8xf32>, 2>
// CHECK-NEXT: return
}

// -----
#map2 = (d0) -> ((d0 * 2048) floordiv 32)

// CHECK: func @loop_dma_dependent
func @loop_dma_dependent(%arg2: memref<512x32xvector<8xf32>>) {
  %num_elts = constant 256 : index
  %c0 = constant 0 : index
  %0 = alloc() : memref<64x4xvector<8xf32>, 2>
  %1 = alloc() : memref<64x4xvector<8xf32>, 2>
  %2 = alloc() : memref<64x4xvector<8xf32>, 2>
  %3 = alloc() : memref<2xi32>
  %4 = alloc() : memref<2xi32>
  %5 = alloc() : memref<2xi32>

  // The two DMAs below are dependent (incoming and outgoing on the same
  // memref) in the same iteration; so no pipelining here.
  // CHECK-NOT: affine.dma_start
  // CHECK: affine.for %{{.*}} = 0 to 8 {
  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    affine.dma_start %arg2[%6, %c0], %2[%c0, %c0], %5[%c0], %num_elts : memref<512x32xvector<8xf32>>, memref<64x4xvector<8xf32>, 2>, memref<2xi32>
    affine.dma_wait %5[%c0], %num_elts : memref<2xi32>

    affine.dma_start %2[%c0, %c0], %arg2[%6, %c0], %5[%c0], %num_elts : memref<64x4xvector<8xf32>, 2>, memref<512x32xvector<8xf32>>, memref<2xi32>
    affine.dma_wait %5[%c0], %num_elts : memref<2xi32>
  }
  dealloc %5 : memref<2xi32>
  dealloc %4 : memref<2xi32>
  dealloc %3 : memref<2xi32>
  dealloc %2 : memref<64x4xvector<8xf32>, 2>
  dealloc %1 : memref<64x4xvector<8xf32>, 2>
  dealloc %0 : memref<64x4xvector<8xf32>, 2>
  return
}

// -----

// CHECK-LABEL: func @escaping_use
func @escaping_use(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: affine.dma_start
  // CHECK: affine.for %{{.*}} = 0 to 16 {
  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    affine.dma_wait %tag[%zero], %num_elt : memref<1 x i32>
    // escaping use; no DMA pipelining / double buffering will be done.
    "foo"(%Av) : (memref<32 x 32 x f32, 2>) -> ()
  }
  dealloc %tag : memref<1 x i32>
  dealloc %Av : memref<32 x 32 x f32, 2>
  return
// CHECK:        "foo"(%{{[0-9]+}}) : (memref<32x32xf32, 2>) -> ()
// CHECK:      }
// CHECK:      return
}

// -----

// CHECK-LABEL: func @escaping_tag
func @escaping_tag(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: affine.dma_start
  // CHECK: affine.for %{{.*}} = 0 to 16 {
  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    affine.dma_wait %tag[%zero], %num_elt : memref<1 x i32>
    // escaping use; no DMA pipelining / double buffering will be done.
    "foo"(%tag) : (memref<1 x i32>) -> ()
  }
  dealloc %tag : memref<1 x i32>
  dealloc %Av : memref<32 x 32 x f32, 2>
  return
// CHECK:        "foo"(%{{[0-9]+}}) : (memref<1xi32>) -> ()
// CHECK:      }
// CHECK:      return
}


// -----

// CHECK-LABEL: func @live_out_use
func @live_out_use(%arg0: memref<512 x 32 x f32>) -> f32 {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: affine.dma_start
  // CHECK: affine.for %{{.*}} = 0 to 16 {
  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    affine.dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  // Use live out of 'affine.for' op; no DMA pipelining will be done.
  %v = affine.load %Av[%zero, %zero] : memref<32 x 32 x f32, 2>
  dealloc %tag : memref<1 x i32>
  dealloc %Av : memref<32 x 32 x f32, 2>
  return %v : f32
// CHECK:      %{{[0-9]+}} = affine.load %{{[0-9]+}}[%{{.*}}, %{{.*}}] : memref<32x32xf32, 2>
// CHECK:      return
}

// -----

// CHECK-LABEL: func @dynamic_shape_dma_buffer
func @dynamic_shape_dma_buffer(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index

  %Av = alloc(%c32, %c32) : memref<? x ? x f32, 2>
  %tag = alloc() : memref<1 x i32>

// Double buffering for dynamic shaped buffer.
// CHECK:       %{{.*}} = alloc(%{{.*}}, %{{.*}}) : memref<?x?xf32, 2>
// CHECK-NEXT:  %{{.*}} = dim %{{.*}}, 0 : memref<?x?xf32, 2>
// CHECK-NEXT:  %{{.*}} = dim %{{.*}}, 1 : memref<?x?xf32, 2>
// CHECK-NEXT:  %{{.*}} = alloc(%{{.*}}, %{{.*}}) : memref<2x?x?xf32, 2>
// CHECK:       affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}} mod 2, 0, 0], %{{.*}}[%{{.*}} mod 2, 0], %{{.*}}
  affine.for %kTT = 0 to 16 {
    affine.dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %tag[%zero], %num_elt :
      memref<512 x 32 x f32>,
      memref<? x ? x f32, 2>, memref<1 x i32>
    affine.dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  dealloc %Av : memref<? x ? x f32, 2>
  return
// CHECK-NEXT:  affine.for %{{.*}} = 1 to 16 {
// CHECK:         affine.dma_start %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}[%{{.*}} mod 2, 0, 0], %{{.*}}[%{{.*}} mod 2, 0], %{{.*}}
// CHECK:         affine.dma_wait %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<2x1xi32>
// CHECK:       }
// CHECK:       affine.dma_wait %{{.*}}[%{{.*}} mod 2, 0], %{{.*}} : memref<2x1xi32>
// CHECK:       return
}

// Memref replacement will fail here due to a non-dereferencing use. However,
// no incorrect transformation is performed in spite of one of the uses being a
// dereferencing one since replaceAllMemRefUsesWith checks for escaping uses
// before performing any replacement.
// CHECK-LABEL: func @escaping_and_indexed_use_mix
func @escaping_and_indexed_use_mix() {
  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>
  %tag = alloc() : memref<1 x f32>
  %zero = constant 0 : index
  %num_elts = constant 32 : index

  // alloc for the buffer is created but no replacement should happen.
  affine.for %i = 0 to 8 {
    affine.dma_start %A[%i], %Ah[%i], %tag[%zero], %num_elts : memref<256 x f32>, memref<32 x f32, 1>, memref<1 x f32>
    affine.dma_wait %tag[%zero], %num_elts : memref<1 x f32>
    "compute"(%Ah) : (memref<32 x f32, 1>) -> ()
    %v = affine.load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    "foo"(%v) : (f32) -> ()
  }
  dealloc %A : memref<256 x f32, (d0) -> (d0), 0>
  dealloc %Ah : memref<32 x f32, (d0) -> (d0), 1>
  return
}
// No replacement.
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK-NEXT:   affine.dma_start %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], %{{.*}}
// CHECK-NEXT:   affine.dma_wait %{{.*}}[%{{.*}}], %{{.*}} : memref<1xf32>
// CHECK-NEXT:   "compute"(%{{.*}}) : (memref<32xf32, 1>) -> ()
// CHECK-NEXT:   [[VAL:%[0-9]+]] = affine.load %{{.*}}[%{{.*}}] : memref<32xf32, 1>
// CHECK-NEXT:   "foo"([[VAL]]) : (f32) -> ()
