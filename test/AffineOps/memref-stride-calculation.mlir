// RUN: mlir-opt %s -test-memref-stride-calculation -o /dev/null | FileCheck %s

func @f(%0: index) {
// CHECK-LABEL: Testing: f
  %1 = alloc() : memref<3x4x5xf32>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %2 = alloc(%0) : memref<3x4x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %3 = alloc(%0) : memref<3x?x5xf32>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %4 = alloc(%0) : memref<?x4x5xf32>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %5 = alloc(%0, %0) : memref<?x4x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %6 = alloc(%0, %0, %0) : memref<?x?x?xf32>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1

  %11 = alloc() : memref<3x4x5xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %b11 = alloc() : memref<3x4x5xf32, offset: 0, strides: [20, 5, 1]>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %12 = alloc(%0) : memref<3x4x?xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %13 = alloc(%0) : memref<3x?x5xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %14 = alloc(%0) : memref<?x4x5xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: 20, 5, 1
  %15 = alloc(%0, %0) : memref<?x4x?xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %16 = alloc(%0, %0, %0) : memref<?x?x?xf32, (i, j, k)->(i, j, k)>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1

  %21 = alloc()[%0] : memref<3x4x5xf32, (i, j, k)[M]->(32 * i + 16 * j + M * k + 1)>
// CHECK: MemRefType offset: 1 strides: 32, 16, ?
  %22 = alloc()[%0] : memref<3x4x5xf32, (i, j, k)[M]->(32 * i + M * j + 16 * k + 3)>
// CHECK: MemRefType offset: 3 strides: 32, ?, 16
  %b22 = alloc(%0)[%0, %0] : memref<3x4x?xf32, offset: 0, strides: [?, ?, 1]>
// CHECK: MemRefType offset: 0 strides: ?, ?, 1
  %23 = alloc(%0)[%0] : memref<3x?x5xf32, (i, j, k)[M]->(M * i + 32 * j + 16 * k + 7)>
// CHECK: MemRefType offset: 7 strides: ?, 32, 16
  %b23 = alloc(%0)[%0] : memref<3x?x5xf32, offset: 0, strides: [?, 5, 1]>
// CHECK: MemRefType offset: 0 strides: ?, 5, 1
  %24 = alloc(%0)[%0] : memref<3x?x5xf32, (i, j, k)[M]->(M * i + 32 * j + 16 * k + M)>
// CHECK: MemRefType offset: ? strides: ?, 32, 16
  %b24 = alloc(%0)[%0, %0] : memref<3x?x5xf32, offset: ?, strides: [?, 32, 16]>
// CHECK: MemRefType offset: ? strides: ?, 32, 16
  %25 = alloc(%0, %0)[%0, %0] : memref<?x?x16xf32, (i, j, k)[M, N]->(M * i + N * j + k + 1)>
// CHECK: MemRefType offset: 1 strides: ?, ?, 1
  %b25 = alloc(%0, %0)[%0, %0] : memref<?x?x16xf32, offset: 1, strides: [?, ?, 1]>
// CHECK: MemRefType offset: 1 strides: ?, ?, 1
  %26 = alloc(%0)[] : memref<?xf32, (i)[M]->(i)>
// CHECK: MemRefType offset: 0 strides: 1
  %27 = alloc()[%0] : memref<5xf32, (i)[M]->(M)>
// CHECK: MemRefType memref<5xf32, (d0)[s0] -> (s0)> cannot be converted to strided form
  %28 = alloc()[%0] : memref<5xf32, (i)[M]->(123)>
// CHECK: MemRefType memref<5xf32, (d0)[s0] -> (123)> cannot be converted to strided form
  %29 = alloc()[%0] : memref<f32, ()[M]->(M)>
// CHECK: MemRefType offset: ? strides:
  %30 = alloc()[%0] : memref<f32, ()[M]->(123)>
// CHECK: MemRefType offset: 123 strides:

  %100 = alloc(%0, %0)[%0, %0] : memref<?x?x16xf32, (i, j, k)[M, N]->(i + j, j, k), (i, j, k)[M, N]->(M * i + N * j + k + 1)>
// CHECK: MemRefType memref<?x?x16xf32, (d0, d1, d2)[s0, s1] -> (d0 + d1, d1, d2), (d0, d1, d2)[s0, s1] -> (d0 * s0 + d1 * s1 + d2 + 1)> cannot be converted to strided form
  %101 = alloc() : memref<3x4x5xf32, (i, j, k)->(i floordiv 4 + j + k)>
// CHECK: MemRefType memref<3x4x5xf32, (d0, d1, d2) -> (d0 floordiv 4 + d1 + d2)> cannot be converted to strided form
  %102 = alloc() : memref<3x4x5xf32, (i, j, k)->(i ceildiv 4 + j + k)>
// CHECK: MemRefType memref<3x4x5xf32, (d0, d1, d2) -> (d0 ceildiv 4 + d1 + d2)> cannot be converted to strided form
  %103 = alloc() : memref<3x4x5xf32, (i, j, k)->(i mod 4 + j + k)>
// CHECK: MemRefType memref<3x4x5xf32, (d0, d1, d2) -> (d0 mod 4 + d1 + d2)> cannot be converted to strided form
  return
}
