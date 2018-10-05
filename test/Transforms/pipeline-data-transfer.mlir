// RUN: mlir-opt %s -pipeline-data-transfer | FileCheck %s
// CHECK:      #map0 = (d0) -> (d0 mod 2)
// CHECK-NEXT: #map1 = (d0) -> (d0 - 1)
// CHECK-NEXT: mlfunc @loop_nest_dma() {
// CHECK-NEXT:   %c8 = constant 8 : affineint
// CHECK-NEXT:   %c0 = constant 0 : affineint
// CHECK-NEXT:   %0 = alloc() : memref<2x1xf32>
// CHECK-NEXT:   %1 = alloc() : memref<2x32xf32>
// CHECK-NEXT:   %2 = alloc() : memref<256xf32, (d0) -> (d0)>
// CHECK-NEXT:   %3 = alloc() : memref<32xf32, (d0) -> (d0), 1>
// CHECK-NEXT:   %4 = alloc() : memref<1xf32>
// CHECK-NEXT:   %c0_0 = constant 0 : affineint
// CHECK-NEXT:   %c128 = constant 128 : affineint
// CHECK-NEXT:   %5 = affine_apply #map0(%c0)
// CHECK-NEXT:   %6 = affine_apply #map0(%c0)
// CHECK-NEXT:   "dma.in.start"(%2, %c0, %1, %5, %c0, %c128, %0, %6, %c0_0) : (memref<256xf32, (d0) -> (d0)>, affineint, memref<2x32xf32>, affineint, affineint, affineint, memref<2x1xf32>, affineint, affineint) -> ()
// CHECK-NEXT:   for %i0 = 1 to 7 {
// CHECK-NEXT:     %7 = affine_apply #map0(%i0)
// CHECK-NEXT:     %8 = affine_apply #map0(%i0)
// CHECK-NEXT:     "dma.in.start"(%2, %i0, %1, %7, %i0, %c128, %0, %8, %c0_0) : (memref<256xf32, (d0) -> (d0)>, affineint, memref<2x32xf32>, affineint, affineint, affineint, memref<2x1xf32>, affineint, affineint) -> ()
// CHECK-NEXT:     %9 = affine_apply #map1(%i0)
// CHECK-NEXT:     %10 = affine_apply #map0(%9)
// CHECK-NEXT:     %11 = "dma.finish"(%0, %10, %c0_0) : (memref<2x1xf32>, affineint, affineint) -> affineint
// CHECK-NEXT:     %12 = affine_apply #map0(%9)
// CHECK-NEXT:     %13 = load %1[%12, %9] : memref<2x32xf32>
// CHECK-NEXT:     %14 = "compute"(%13) : (f32) -> f32
// CHECK-NEXT:     %15 = affine_apply #map0(%9)
// CHECK-NEXT:     store %14, %1[%15, %9] : memref<2x32xf32>
// CHECK-NEXT:     for %i1 = 0 to 127 {
// CHECK-NEXT:       "do_more_compute"(%9, %i1) : (affineint, affineint) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %16 = affine_apply #map1(%c8)
// CHECK-NEXT:   %17 = affine_apply #map0(%16)
// CHECK-NEXT:   %18 = "dma.finish"(%0, %17, %c0_0) : (memref<2x1xf32>, affineint, affineint) -> affineint
// CHECK-NEXT:   %19 = affine_apply #map0(%16)
// CHECK-NEXT:   %20 = load %1[%19, %16] : memref<2x32xf32>
// CHECK-NEXT:   %21 = "compute"(%20) : (f32) -> f32
// CHECK-NEXT:   %22 = affine_apply #map0(%16)
// CHECK-NEXT:   store %21, %1[%22, %16] : memref<2x32xf32>
// CHECK-NEXT:   for %i2 = 0 to 127 {
// CHECK-NEXT:     "do_more_compute"(%16, %i2) : (affineint, affineint) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
mlfunc @loop_nest_dma() {

  %A = alloc() : memref<256 x f32, (d0) -> (d0), 0>
  %Ah = alloc() : memref<32 x f32, (d0) -> (d0), 1>

  %tag = alloc() : memref<1 x f32>

  %zero = constant 0 : affineint
  %size = constant 128 : affineint

  for %i = 0 to 7 {
    "dma.in.start"(%A, %i, %Ah, %i, %size, %tag, %zero) : (memref<256 x f32, (d0)->(d0), 0>, affineint, memref<32 x f32, (d0)->(d0), 1>, affineint, affineint, memref<1 x f32>, affineint) -> ()
    "dma.finish"(%tag, %zero) : (memref<1 x f32>, affineint) -> affineint
    %v = load %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    %r = "compute"(%v) : (f32) -> (f32)
    store %r, %Ah[%i] : memref<32 x f32, (d0) -> (d0), 1>
    for %j = 0 to 127 {
      "do_more_compute"(%i, %j) : (affineint, affineint) -> ()
    }
  }
  return
}
