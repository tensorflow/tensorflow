// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s

// CHECK:      #map0 = (d0) -> (d0 mod 2, d0 mod 2)
// CHECK-NEXT: #map1 = (d0) -> (d0 - 1)
// CHECK-NEXT: #map2 = (d0) -> (d0 mod 2)
// CHECK-NEXT: mlfunc @loop_nest_dma() {
// CHECK-NEXT:   %c8 = constant 8 : index
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %0 = alloc() : memref<2x1xf32>
// CHECK-NEXT:   %1 = alloc() : memref<2x32xf32>
// CHECK-NEXT:   %2 = alloc() : memref<256xf32, (d0) -> (d0)>
// CHECK-NEXT:   %3 = alloc() : memref<32xf32, (d0) -> (d0), 1>
// CHECK-NEXT:   %4 = alloc() : memref<1xf32>
// CHECK-NEXT:   %c0_0 = constant 0 : index
// CHECK-NEXT:   %c128 = constant 128 : index
// CHECK-NEXT:   %5 = affine_apply #map0(%c0)
// CHECK-NEXT:   dma_start %2[%c0], %1[%5#0, %c0], %c128, %0[%5#1, %c0_0] : memref<256xf32, (d0) -> (d0)>, memref<2x32xf32>, memref<2x1xf32>
// CHECK-NEXT:   for %i0 = 1 to 7 {
// CHECK-NEXT:     %6 = affine_apply #map0(%i0)
// CHECK-NEXT:     dma_start %2[%i0], %1[%6#0, %i0], %c128, %0[%6#1, %c0_0] : memref<256xf32, (d0) -> (d0)>, memref<2x32xf32>, memref<2x1xf32>
// CHECK-NEXT:     %7 = affine_apply #map1(%i0)
// CHECK-NEXT:     %8 = affine_apply #map2(%7)
// CHECK-NEXT:     %9 = affine_apply #map2(%7)
// CHECK-NEXT:     dma_wait %0[%8, %c0_0] : memref<2x1xf32>
// CHECK-NEXT:    %10 = load %1[%9, %7] : memref<2x32xf32>
// CHECK-NEXT:     %11 = "compute"(%10) : (f32) -> f32
// CHECK-NEXT:     store %11, %1[%9, %7] : memref<2x32xf32>
// CHECK-NEXT:     for %i1 = 0 to 127 {
// CHECK-NEXT:       "do_more_compute"(%7, %i1) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %12 = affine_apply #map1(%c8)
// CHECK-NEXT:   %13 = affine_apply #map2(%12)
// CHECK-NEXT:   %14 = affine_apply #map2(%12)
// CHECK-NEXT:   dma_wait %0[%13, %c0_0] : memref<2x1xf32>
// CHECK-NEXT:   %15 = load %1[%14, %12] : memref<2x32xf32>
// CHECK-NEXT:   %16 = "compute"(%15) : (f32) -> f32
// CHECK-NEXT:   store %16, %1[%14, %12] : memref<2x32xf32>
// CHECK-NEXT:   for %i2 = 0 to 127 {
// CHECK-NEXT:     "do_more_compute"(%12, %i2) : (index, index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
mlfunc @loop_nest_dma() {

  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>

  %tag = alloc() : memref<1 x f32>

  %zero = constant 0 : index
  %size = constant 128 : index

  for %i = 0 to 7 {
    dma_start %A[%i], %Ah[%i], %size, %tag[%zero] : memref<256 x f32, (d0)->(d0), 0>, memref<32 x f32, (d0)->(d0), 1>, memref<1 x f32>
    dma_wait %tag[%zero] : memref<1 x f32>
    %v = load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    %r = "compute"(%v) : (f32) -> (f32)
    store %r, %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    for %j = 0 to 127 {
      "do_more_compute"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
