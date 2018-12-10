// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s

// CHECK-LABEL: mlfunc @loop_nest_dma() {
mlfunc @loop_nest_dma() {
// CHECK:        %0 = alloc() : memref<256xf32>
// CHECK:        %1 = alloc() : memref<2x32xf32, 1>
// CHECK:        %2 = alloc() : memref<2x1xf32>
// CHECK:        dma_start %0[%c0], %1[%3#0, %c0], %c128, %2[%3#1, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:   for %i0 = 1 to 8 {
// CHECK-NEXT:     %4 = affine_apply #map0(%i0)
// CHECK-NEXT:     dma_start %0[%i0], %1[%4#0, %i0], %c128, %2[%4#1, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:     %5 = affine_apply #map1(%i0)
// CHECK-NEXT:     %6 = affine_apply #map2(%5)
// CHECK-NEXT:     %7 = affine_apply #map2(%5)
// CHECK-NEXT:     dma_wait %2[%6, %c0_0], %c128 : memref<2x1xf32>
// CHECK-NEXT:     %8 = load %1[%7, %5] : memref<2x32xf32, 1>
// CHECK-NEXT:     %9 = "compute"(%8) : (f32) -> f32
// CHECK-NEXT:     store %9, %1[%7, %5] : memref<2x32xf32, 1>
// CHECK-NEXT:     for %i1 = 0 to 128 {
// CHECK-NEXT:       "do_more_compute"(%5, %i1) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %10 = affine_apply #map1(%c8)
// CHECK-NEXT:   %11 = affine_apply #map2(%10)
// CHECK-NEXT:   %12 = affine_apply #map2(%10)
// CHECK-NEXT:   dma_wait %2[%11, %c0_0], %c128 : memref<2x1xf32>
// CHECK-NEXT:   %13 = load %1[%12, %10] : memref<2x32xf32, 1>
// CHECK-NEXT:   %14 = "compute"(%13) : (f32) -> f32
// CHECK-NEXT:   store %14, %1[%12, %10] : memref<2x32xf32, 1>
// CHECK-NEXT:   for %i2 = 0 to 128 {
// CHECK-NEXT:     "do_more_compute"(%10, %i2) : (index, index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return

  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>

  %tag = alloc() : memref<1 x f32>

  %zero = constant 0 : index
  %num_elts = constant 128 : index

  for %i = 0 to 8 {
    dma_start %A[%i], %Ah[%i], %num_elts, %tag[%zero] : memref<256 x f32>, memref<32 x f32, 1>, memref<1 x f32>
    dma_wait %tag[%zero], %num_elts : memref<1 x f32>
    %v = load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    %r = "compute"(%v) : (f32) -> (f32)
    store %r, %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    for %j = 0 to 128 {
      "do_more_compute"(%i, %j) : (index, index) -> ()
    }
  }
  return
}

#map0 = (d0, d1) -> (d0, d1)
#map1 = (d0, d1) -> ((d0 * 2048 + d1 * 256) floordiv 32, 0)
#map2 = (d0) -> ((d0 * 2048) floordiv 32, 0)
// CHECK: mlfunc @loop_dma_nested(%arg0 : memref<512x32xvector<8xf32>
mlfunc @loop_dma_nested(%arg0 : memref<512x32xvector<8xf32>, #map0>, %arg1 : memref<512x32xvector<8xf32>, #map0>, %arg2 : memref<512x32xvector<8xf32>, #map0>) {
  %num_elts = constant 256 : index
  %c0 = constant 0 : index
  %0 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %1 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %2 = alloc() : memref<64x4xvector<8xf32>, #map0, 2>
  %3 = alloc() : memref<2xi32>
  %4 = alloc() : memref<2xi32>
  %5 = alloc() : memref<2xi32>
  // Prologue for DMA overlap on arg2.
  // CHECK: dma_start %arg2[
  // CHECK-NEXT: for %i0 = 1 to 8 {
  for %i0 = 0 to 8 {
    %6 = affine_apply #map2(%i0)
    dma_start %arg2[%6#0, %6#1], %2[%c0, %c0], %num_elts, %5[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>
    // Steady state for DMA overlap on arg2
    // CHECK: dma_start %arg2[
    // CHECK: dma_wait %1[
    // Prologue for DMA overlap on arg0, arg1 nested within i0
    // CHECK: dma_start %arg0[
    // CHECK: dma_start %arg1[
    // CHECK-NEXT for %i1 = 1 to 8 {
    for %i1 = 0 to 8 {
      %7 = affine_apply #map1(%i0, %i1)
      %8 = affine_apply #map2(%i1)
      dma_start %arg0[%7#0, %7#1], %0[%c0, %c0], %num_elts, %3[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
      dma_start %arg1[%8#0, %8#1], %1[%c0, %c0], %num_elts, %4[%c0] : memref<512x32xvector<8xf32>, #map0>, memref<64x4xvector<8xf32>, #map0, 2>, memref<2xi32>
      dma_wait %3[%c0], %num_elts : memref<2xi32>
      dma_wait %4[%c0], %num_elts : memref<2xi32>
      // Steady state for DMA overlap on arg0, arg1
      // CHECK: dma_start %arg0[
      // CHECK: dma_start %arg1[
      // CHECK: dma_wait %10[
      // CHECK: dma_wait %11[
      // CHECK-NEXT: for %i2 = 0 to 4 {
      for %i2 = 0 to 4 {
        "foo"() : () -> ()
      }
    }
    // epilogue for arg0, arg1
    // CHECK: dma_wait %10[
    // CHECK: dma_wait %11[

    // epilogue for DMA overlap on %arg2
    // CHECK:  dma_wait %1[%31, %c0_2], %c256 : memref<2x2xi32>
    // Within the epilogue for arg2's DMA, we have the DMAs on %arg1, %arg2 nested.
    // CHECK:  dma_start %arg0[
    // CHECK:  dma_start %arg1[
    // CHECK:  for %i4 = 1 to 8 {
    // CHECK:    dma_start %arg0[
    // CHECK:    dma_start %arg1[
    // CHECK:    dma_wait %36[
    // CHECK:    dma_wait %37[
    // CHECK:    for %i5 = 0 to 4 {
    // CHECK:      "foo"() : () -> ()
    // CHECK:  dma_wait %36[
    // CHECK:  dma_wait %37[
    // CHECK:  for %i6 = 0 to 4 {

    // The DMAs below are outgoing DMAs on arg2, not yet overlapped.
    // CHECK: dma_start %0{{.*}}, %arg2[
    // CHECK-NEXT:  dma_wait %1[
    dma_start %2[%c0, %c0], %arg2[%6#0, %6#1], %num_elts, %5[%c0] : memref<64x4xvector<8xf32>, #map0, 2>, memref<512x32xvector<8xf32>, #map0>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>
  } // CHECK }
  return
}

// CHECK-LABEL: mlfunc @escaping_use
mlfunc @escaping_use(%arg0: memref<512 x 32 x f32>) {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: dma_start
  // CHECK: for %i0 = 0 to 16 {
  for %kTT = 0 to 16 {
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

// CHECK-LABEL: mlfunc @live_out_use
mlfunc @live_out_use(%arg0: memref<512 x 32 x f32>) -> f32 {
  %c32 = constant 32 : index
  %num_elt = constant 512 : index
  %zero = constant 0 : index
  %Av = alloc() : memref<32 x 32 x f32, 2>
  %tag = alloc() : memref<1 x i32>

  // CHECK-NOT: dma_start
  // CHECK: for %i0 = 0 to 16 {
  for %kTT = 0 to 16 {
    dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %num_elt, %tag[%zero] :
      memref<512 x 32 x f32>,
      memref<32 x 32 x f32, 2>, memref<1 x i32>
    dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  // Use live out of 'for' stmt; no DMA pipelining will be done.
  %v = load %Av[%zero, %zero] : memref<32 x 32 x f32, 2>
  return %v : f32
// CHECK:      %{{[0-9]+}} = load %{{[0-9]+}}[%c0, %c0] : memref<32x32xf32, 2>
// CHECK-NEXT: return
}

// CHECK-LABEL: mlfunc @dynamic_shape_dma_buffer
mlfunc @dynamic_shape_dma_buffer(%arg0: memref<512 x 32 x f32>) {
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

// CHECK:  dma_start %arg0[%c0_0, %c0_0], %3[%5#0, %c0_0, %c0_0],
// CHECK-NEXT:  for %i0 = 1 to 16 {
  for %kTT = 0 to 16 {
    dma_start %arg0[%zero, %zero], %Av[%zero, %zero], %num_elt, %tag[%zero] :
      memref<512 x 32 x f32>,
      memref<? x ? x f32, 2>, memref<1 x i32>
    dma_wait %tag[%zero], %num_elt : memref<1 x i32>
  }
  return
// CHECK:          dma_start %arg0[%c0_0, %c0_0], %3[%6#0, %c0_0, %c0_0], %c512, %4[%6#1, %c0_0]
// CHECK:          dma_wait %4[%8, %c0_0], %c512 : memref<2x1xi32>
// CHECK:       }
// CHECK:       dma_wait %4[%11, %c0_0], %c512 : memref<2x1xi32>
// CHECK-NEXT:  return
}
