// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s

// CHECK-DAG: [[MOD_2:#map[0-9]+]] = (d0) -> (d0 mod 2)
// CHECK-DAG: [[FLOOR_MOD_2:#map[0-9]+]] = (d0) -> ((d0 floordiv 4) mod 2)
// CHECK-DAG: [[REMAP_SHIFT_MINUS_4:#map[0-9]+]] = (d0) -> (d0 - 4)
// CHECK-DAG: [[MAP_MINUS_1:#map[0-9]+]] = (d0) -> (d0 - 1)

// CHECK-LABEL: func @loop_nest_dma() {
func @loop_nest_dma() {

  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>

  %tag = alloc() : memref<1 x f32>

  %zero = constant 0 : index
  %num_elts = constant 128 : index

  affine.for %i = 0 to 8 {
    dma_start %A[%i], %Ah[%i], %num_elts, %tag[%zero] : memref<256 x f32>, memref<32 x f32, 1>, memref<1 x f32>
    dma_wait %tag[%zero], %num_elts : memref<1 x f32>
    %v = load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    %r = "compute"(%v) : (f32) -> (f32)
    store %r, %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    affine.for %j = 0 to 128 {
      "do_more_compute"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
// CHECK:       %0 = alloc() : memref<256xf32>
// CHECK:       %1 = alloc() : memref<2x32xf32, 1>
// CHECK-NEXT:  %2 = alloc() : memref<2x1xf32>
// CHECK-NEXT:  %3 = affine.apply [[MOD_2]](%c0)
// CHECK-NEXT:  %4 = affine.apply [[MOD_2]](%c0)
// CHECK-NEXT:  dma_start %0[%c0], %1[%3, %c0], %c128, %2[%4, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:  affine.for %i0 = 1 to 8 {
// CHECK-NEXT:    %5 = affine.apply [[MOD_2]](%i0)
// CHECK-NEXT:    %6 = affine.apply [[MOD_2]](%i0)
// CHECK-NEXT:    dma_start %0[%i0], %1[%5, %i0], %c128, %2[%6, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:    %7 = affine.apply [[MAP_MINUS_1]](%i0)
// CHECK-NEXT:    %8 = affine.apply [[MOD_2]](%7)
// CHECK-NEXT:    %9 = affine.apply [[MOD_2]](%7)
// CHECK-NEXT:    dma_wait %2[%8, %c0_0], %c128 : memref<2x1xf32>
// CHECK-NEXT:    %10 = load %1[%9, %7] : memref<2x32xf32, 1>
// CHECK-NEXT:    %11 = "compute"(%10) : (f32) -> f32
// CHECK-NEXT:    store %11, %1[%9, %7] : memref<2x32xf32, 1>
// CHECK-NEXT:    affine.for %i1 = 0 to 128 {
// CHECK-NEXT:      "do_more_compute"(%7, %i1) : (index, index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %12 = affine.apply [[MAP_MINUS_1]](%c8)
// CHECK-NEXT:  %13 = affine.apply [[MOD_2]](%12)
// CHECK-NEXT:  %14 = affine.apply [[MOD_2]](%12)
// CHECK-NEXT:  dma_wait %2[%13, %c0_0], %c128 : memref<2x1xf32>
// CHECK-NEXT:  %15 = load %1[%14, %12] : memref<2x32xf32, 1>
// CHECK-NEXT:  %16 = "compute"(%15) : (f32) -> f32
// CHECK-NEXT:  store %16, %1[%14, %12] : memref<2x32xf32, 1>
// CHECK-NEXT:  affine.for %i2 = 0 to 128 {
// CHECK-NEXT:    "do_more_compute"(%12, %i2) : (index, index) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}


// CHECK-LABEL: @loop_step
func @loop_step(%arg0: memref<512xf32>,
                  %arg1: memref<512xf32>) {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  affine.for %i0 = 0 to 512 step 4 {
    %1 = alloc() : memref<4xf32, 1>
    %2 = alloc() : memref<1xi32>
    dma_start %arg0[%i0], %1[%c0], %c4, %2[%c0]
              : memref<512xf32>, memref<4xf32, 1>, memref<1xi32>
    dma_wait %2[%c0], %c4 : memref<1xi32>
    "compute"(%i0) : (index) -> ()
  }
  return
}
// CHECK:        [[TAG:%[0-9]+]] = alloc() : memref<2x1xi32>
// CHECK:        %2 = affine.apply [[FLOOR_MOD_2]](%c0)
// CHECK:        %3 = affine.apply [[FLOOR_MOD_2]](%c0)
// CHECK-NEXT:   dma_start %arg0[%c0], %0[%2, %c0_0], %c4, [[TAG]][%3, %c0_0] : memref<512xf32>, memref<2x4xf32, 1>, memref<2x1xi32>
// CHECK-NEXT:   affine.for %i0 = 4 to 512 step 4 {
// CHECK-NEXT:     %4 = affine.apply [[FLOOR_MOD_2]](%i0)
// CHECK-NEXT:     %5 = affine.apply [[FLOOR_MOD_2]](%i0)
// CHECK-NEXT:     dma_start %arg0[%i0], %0[%4, %c0_0], %c4, [[TAG]][%5, %c0_0] : memref<512xf32>, memref<2x4xf32, 1>, memref<2x1xi32>
// CHECK-NEXT:     %6 = affine.apply [[REMAP_SHIFT_MINUS_4]](%i0)
// CHECK-NEXT:     %7 = affine.apply [[FLOOR_MOD_2]](%6)
// CHECK:          dma_wait [[TAG]][%7, %c0_0], %c4 : memref<2x1xi32>
// CHECK-NEXT:     "compute"(%6) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   [[SHIFTED:%[0-9]+]] = affine.apply [[REMAP_SHIFT_MINUS_4]](%c512)
// CHECK-NEXT:   %10 = affine.apply [[FLOOR_MOD_2]]([[SHIFTED]])
// CHECK:        dma_wait [[TAG]][%10, %c0_0], %c4 : memref<2x1xi32>
// CHECK-NEXT:   "compute"(%9) : (index) -> ()
// CHECK-NEXT:   return
// CHECK-NEXT: }

#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0, d1) -> ((d0 * 2048 + d1 * 256) floordiv 32)
#map2 = (d0) -> ((d0 * 2048) floordiv 32)
// CHECK-LABEL: func @loop_dma_nested(%arg0: memref<512x32xvector<8xf32>
func @loop_dma_nested(%arg0: memref<512x32xvector<8xf32>, #map0>, %arg1: memref<512x32xvector<8xf32>, #map0>, %arg2: memref<512x32xvector<8xf32>, #map0>) {
  %num_elts = constant 256 : index
  %c0 = constant 0 : index
  %0 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %1 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %2 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %3 = alloc() : memref<2xi32>
  %4 = alloc() : memref<2xi32>
  %5 = alloc() : memref<2xi32>
  // Prologue for DMA overlap on arg2.
  // CHECK:[[TAG_ARG2:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK: dma_start %arg2[
  // CHECK: affine.for %i0 = 1 to 8 {
  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    dma_start %arg2[%6, %c0], %2[%c0, %c0], %num_elts, %5[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>
    // Steady state for DMA overlap on arg2
    // CHECK: dma_start %arg2[
    // CHECK: dma_wait [[TAG_ARG2]]
    // Prologue for DMA overlap on arg0, arg1 nested within i0
    // CHECK: [[TAG_ARG0:%[0-9]+]] = alloc() : memref<2x2xi32>
    // CHECK: [[TAG_ARG1:%[0-9]+]] = alloc() : memref<2x2xi32>
    // CHECK: dma_start %arg0[
    // CHECK: dma_start %arg1[
    // CHECK-NEXT affine.for %i1 = 1 to 8 {
    affine.for %i1 = 0 to 8 {
      %7 = affine.apply #map1(%i0, %i1)
      %8 = affine.apply #map2(%i1)
      dma_start %arg0[%7, %c0], %0[%c0, %c0], %num_elts, %3[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
      dma_start %arg1[%8, %c0], %1[%c0, %c0], %num_elts, %4[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
      dma_wait %3[%c0], %num_elts : memref<2xi32>
      dma_wait %4[%c0], %num_elts : memref<2xi32>
      // Steady state for DMA overlap on arg0, arg1
      // CHECK: dma_start %arg0[
      // CHECK: dma_start %arg1[
      // CHECK: dma_wait [[TAG_ARG0]]
      // CHECK: dma_wait [[TAG_ARG1]]
      // CHECK-NEXT: affine.for %i2 = 0 to 4 {
      affine.for %i2 = 0 to 4 {
        "foo"() : () -> ()
      }
    }
    // epilogue for arg0, arg1
    // CHECK: dma_wait [[TAG_ARG0]]
    // CHECK: dma_wait [[TAG_ARG1]]
  // epilogue for DMA overlap on %arg2
  // CHECK:  dma_wait [[TAG_ARG2]]
  // Within the epilogue for arg2's DMA, we have the DMAs on %arg1, %arg2 nested.
  // CHECK: [[TAG_ARG0_NESTED:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK: [[TAG_ARG1_NESTED:%[0-9]+]] = alloc() : memref<2x2xi32>
  // CHECK:  dma_start %arg0[
  // CHECK:  dma_start %arg1[
  // CHECK:  affine.for %i4 = 1 to 8 {
  // CHECK:    dma_start %arg0[
  // CHECK:    dma_start %arg1[
  // CHECK:    dma_wait [[TAG_ARG0_NESTED]]
  // CHECK:    dma_wait [[TAG_ARG1_NESTED]]
  // CHECK:    affine.for %i5 = 0 to 4 {
  // CHECK:      "foo"() : () -> ()
  // CHECK:  dma_wait [[TAG_ARG0_NESTED]]
  // CHECK:  dma_wait [[TAG_ARG1_NESTED]]
  // CHECK:  affine.for %i6 = 0 to 4 {
  }
  return
// CHECK: }
// CHECK-NEXT: return
}

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
  // CHECK-NOT: dma_start
  // CHECK: affine.for %i0 = 0 to 8 {
  affine.for %i0 = 0 to 8 {
    %6 = affine.apply #map2(%i0)
    dma_start %arg2[%6, %c0], %2[%c0, %c0], %num_elts, %5[%c0] : memref<512x32xvector<8xf32>>, memref<64x4xvector<8xf32>, 2>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>

    dma_start %2[%c0, %c0], %arg2[%6, %c0], %num_elts, %5[%c0] : memref<64x4xvector<8xf32>, 2>, memref<512x32xvector<8xf32>>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>
  } // CHECK: }
  return // CHECK-NEXT: return
}

// CHECK-LABEL: func @escaping_use
func @escaping_use(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: dma_start
  // CHECK: affine.for %i0 = 0 to 16 {
  affine.for %kTT = 0 to 16 {
    dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %num_elt, %tag[%zero] :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    dma_wait %tag[%zero], %num_elt : memref<1 x i32>
    // escaping use; no DMA pipelining / double buffering will be done.
    "foo"(%Av) : (memref<32 x 32 x f32, 2>) -> ()
  }
  return
// CHECK:        "foo"(%{{[0-9]+}}) : (memref<32x32xf32, 2>) -> ()
// CHECK:      }
// CHECK-NEXT: return
}

// CHECK-LABEL: func @live_out_use
func @live_out_use(%arg0: memref<512 x 32 x f32>) -> f32 {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: dma_start
  // CHECK: affine.for %i0 = 0 to 16 {
  affine.for %kTT = 0 to 16 {
    dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %num_elt, %tag[%zero] :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  // Use live out of 'affine.for' inst; no DMA pipelining will be done.
  %v = load %Av[%zero, %zero] : memref<32 x 32 x f32, 2>
  return %v : f32
// CHECK:      %{{[0-9]+}} = load %{{[0-9]+}}[%c0, %c0] : memref<32x32xf32, 2>
// CHECK-NEXT: return
}

// CHECK-LABEL: func @dynamic_shape_dma_buffer
func @dynamic_shape_dma_buffer(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index

  %Av = alloc(%c32, %c32) : memref<? x ? x f32, 2>
  %tag = alloc() : memref<1 x i32>

// Double buffering for dynamic shaped buffer.
// CHECK:       %0 = alloc(%c32, %c32) : memref<?x?xf32, 2>
// CHECK-NEXT:  %1 = dim %0, 0 : memref<?x?xf32, 2>
// CHECK-NEXT:  %2 = dim %0, 1 : memref<?x?xf32, 2>
// CHECK-NEXT:  %3 = alloc(%1, %2) : memref<2x?x?xf32, 2>
// CHECK:       %5 = affine.apply [[MOD_2]](%c0)
// CHECK:       %6 = affine.apply [[MOD_2]](%c0)
// CHECK:       dma_start %arg0[%c0_0, %c0_0], %3[%5, %c0_0, %c0_0], %c512, %4[%6, %c0_0]
  affine.for %kTT = 0 to 16 {
    dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %num_elt, %tag[%zero] :
      memref<512 x 32 x f32>,
      memref<? x ? x f32, 2>, memref<1 x i32>
    dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  return
// CHECK-NEXT:  affine.for %i0 = 1 to 16 {
// CHECK:         %7 = affine.apply [[MOD_2]](%i0)
// CHECK:         %8 = affine.apply [[MOD_2]](%i0)
// CHECK:         dma_start %arg0[%c0_0, %c0_0], %3[%7, %c0_0, %c0_0], %c512, %4[%8, %c0_0]
// CHECK:         dma_wait %4[%10, %c0_0], %c512 : memref<2x1xi32>
// CHECK:       }
// CHECK:       dma_wait %4[%13, %c0_0], %c512 : memref<2x1xi32>
// CHECK-NEXT:  return
}
