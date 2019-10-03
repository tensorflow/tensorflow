// RUN: mlir-opt %s -linalg-fusion | FileCheck %s

#map0 = (d0) -> (d0 + 20)
#map1 = (d0) -> (d0 + 40)
#map2 = (d0) -> (d0 + 30)
#map3 = (d0) -> (d0 + 2)
#map4 = (d0) -> (d0 + 4)
#map5 = (d0) -> (d0 + 3)

func @f1(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>, %B: memref<?x?xf32, offset: ?, strides: [?, 1]>, %C: memref<?x?xf32, offset: ?, strides: [?, 1]>, %D: memref<?x?xf32, offset: ?, strides: [?, 1]>, %E: memref<?x?xf32, offset: ?, strides: [?, 1]>) -> memref<?x?xf32, offset: ?, strides: [?, 1]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %c40 = constant 40 : index
  %c30 = constant 30 : index
  %c20 = constant 20 : index
  %0 = dim %C, 0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = dim %C, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  loop.for %arg5 = %c0 to %0 step %c20 {
    loop.for %arg6 = %c0 to %2 step %c30 {
      loop.for %arg7 = %c0 to %1 step %c40 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
        %9 = dim %5, 0 : memref<?x?xf32, offset: ?, strides: [?, 1]>
        %10 = dim %5, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
        %11 = dim %7, 1 : memref<?x?xf32, offset: ?, strides: [?, 1]>
        loop.for %arg8 = %c0 to %9 step %c2 {
          loop.for %arg9 = %c0 to %11 step %c3 {
            loop.for %B0 = %c0 to %10 step %c4 {
              %12 = affine.apply #map3(%arg8)
              %13 = affine.apply #map4(%B0)
              %14 = linalg.subview %5[%arg8, %12, %c1, %B0, %13, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
              %15 = affine.apply #map5(%arg9)
              %16 = linalg.subview %7[%B0, %13, %c1, %arg9, %15, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
              %17 = linalg.subview %8[%arg8, %12, %c1, %arg9, %15, %c1] : memref<?x?xf32, offset: ?, strides: [?, 1]>
              linalg.matmul(%14, %16, %17) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
            }
          }
        }
      }
    }
  }
  return %E : memref<?x?xf32, offset: ?, strides: [?, 1]>
}
// CHECK-LABEL: func @f1
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//      CHECK: loop.for
//      CHECK:   loop.for
//      CHECK:     loop.for
//      CHECK:      loop.for
//      CHECK:        loop.for
//      CHECK:          loop.for
//      CHECK:            linalg.matmul
//      CHECK:            linalg.matmul
