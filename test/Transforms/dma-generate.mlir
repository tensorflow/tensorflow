// RUN: mlir-opt %s -dma-generate | FileCheck %s

// Index of the buffer for the second DMA is remapped.
// CHECK-DAG: [[MAP:#map[0-9]+]] = (d0) -> (d0 - 256)

// CHECK-LABEL: mlfunc @loop_tiling() {
mlfunc @loop_tiling() {
  %A = alloc() : memref<256 x f32>
  %B = alloc() : memref<512 x f32>
  %F = alloc() : memref<128 x f32, 1>
  // First DMA buffer.
  // CHECK:  %3 = alloc() : memref<256xf32, 1>
  // Tag for first DMA.
  // CHECK:  %4 = alloc() : memref<1xi32>
  // First DMA transfer.
  // CHECK:  dma_start %3[%5], %3[%c0], %c256, %4[%c0] : memref<256xf32, 1>, memref<256xf32, 1>, memref<1xi32>
  // CHECK:  dma_wait %4[%c0], %c256 : memref<1xi32>
  // Second DMA buffer.
  // CHECK:  %6 = alloc() : memref<256xf32, 1>
  // Tag for second DMA.
  // CHECK:  %7 = alloc() : memref<1xi32>
  // Second DMA transfer.
  // CHECK:       dma_start %6[%8], %6[%c0_1], %c256_3, %7[%c0_1] : memref<256xf32, 1>, memref<256xf32, 1>, memref<1xi32>
  // CHECK-NEXT:  dma_wait %7[%c0_1], %c256_3 : memref<1xi32>
  // CHECK: for %i0 = 0 to 256 {
      // CHECK:      %9 = affine_apply #map{{[0-9]+}}(%i0)
      // CHECK-NEXT: %10 = load %3[%9] : memref<256xf32, 1>
      // CHECK:      %11 = affine_apply #map{{[0-9]+}}(%i0)
      // CHECK:      %12 = affine_apply [[MAP]](%11)
      // CHECK-NEXT: %13 = load %6[%12] : memref<256xf32, 1>
      // Already in faster memory space.
      // CHECK:     %14 = load %2[%i0] : memref<128xf32, 1>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  for %i = 0 to 256 {
    load %A[%i] : memref<256 x f32>
    %idx = affine_apply (d0) -> (d0 + 256)(%i)
    load %B[%idx] : memref<512 x f32>
    load %F[%i] : memref<128 x f32, 1>
  }
  return
}
