// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s

// CHECK-LABEL: mlfunc @loop_nest_dma() {
mlfunc @loop_nest_dma() {
// CHECK:        %c8 = constant 8 : index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %0 = alloc() : memref<2x1xf32>
// CHECK-NEXT:   %1 = alloc() : memref<2x32xf32, 1>
// CHECK-NEXT:   %2 = alloc() : memref<256xf32>
// CHECK-NEXT:   %c0_0 = constant 0 : index
// CHECK-NEXT:   %c128 = constant 128 : index
// CHECK-NEXT:   %3 = affine_apply #map0(%c0)
// CHECK-NEXT:   dma_start %2[%c0], %1[%3#0, %c0], %c128, %0[%3#1, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:   for %i0 = 1 to 8 {
// CHECK-NEXT:     %4 = affine_apply #map0(%i0)
// CHECK-NEXT:     dma_start %2[%i0], %1[%4#0, %i0], %c128, %0[%4#1, %c0_0] : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
// CHECK-NEXT:     %5 = affine_apply #map1(%i0)
// CHECK-NEXT:     %6 = affine_apply #map2(%5)
// CHECK-NEXT:     %7 = affine_apply #map2(%5)
// CHECK-NEXT:     dma_wait %0[%6, %c0_0], %c128 : memref<2x1xf32>
// CHECK-NEXT:    %8 = load %1[%7, %5] : memref<2x32xf32, 1>
// CHECK-NEXT:     %9 = "compute"(%8) : (f32) -> f32
// CHECK-NEXT:     store %9, %1[%7, %5] : memref<2x32xf32, 1>
// CHECK-NEXT:     for %i1 = 0 to 128 {
// CHECK-NEXT:       "do_more_compute"(%5, %i1) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %10 = affine_apply #map1(%c8)
// CHECK-NEXT:   %11 = affine_apply #map2(%10)
// CHECK-NEXT:   %12 = affine_apply #map2(%10)
// CHECK-NEXT:   dma_wait %0[%11, %c0_0], %c128 : memref<2x1xf32>
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
    // CHECK: dma_wait %0[
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
      // CHECK: dma_wait %3[
      // CHECK: dma_wait %2[
      // CHECK-NEXT: for %i2 = 0 to 4 {
      for %i2 = 0 to 4 {
        "foo"() : () -> ()
      }
    }
    // epilogue for arg0, arg1
    // CHECK: dma_wait %3[
    // CHECK: dma_wait %2[

    // epilogue for DMA overlap on %arg2
    // CHECK:  dma_wait %0[%31, %c0_2], %c256 : memref<2x2xi32>
    // Within the epilogue for arg2's DMA, we have the DMAs on %arg1, %arg2 nested.
    // CHECK:  dma_start %arg0[
    // CHECK:  dma_start %arg1[
    // CHECK:  for %i4 = 1 to 8 {
    // CHECK:    dma_start %arg0[
    // CHECK:    dma_start %arg1[
    // CHECK:    dma_wait %3[
    // CHECK:    dma_wait %2[
    // CHECK:    for %i5 = 0 to 4 {
    // CHECK:      "foo"() : () -> ()
    // CHECK:  dma_wait %3[
    // CHECK:  dma_wait %2[
    // CHECK:  for %i6 = 0 to 4 {

    // The DMAs below are outgoing DMAs on arg2, not yet overlapped.
    // CHECK: dma_start %1{{.*}}, %arg2[
    // CHECK-NEXT:  dma_wait %0[
    dma_start %2[%c0, %c0], %arg2[%6#0, %6#1], %num_elts, %5[%c0] : memref<64x4xvector<8xf32>, #map0, 2>, memref<512x32xvector<8xf32>, #map0>, memref<2xi32>
    dma_wait %5[%c0], %num_elts : memref<2xi32>
  } // CHECK }
  return
}
