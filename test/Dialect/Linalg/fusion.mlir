// RUN: mlir-opt %s -linalg-fusion | FileCheck %s

#map0 = (d0) -> (d0 + 2)
#map1 = (d0) -> (d0 + 4)
#map2 = (d0) -> (d0 + 3)

// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)

func @f1(%A: memref<?x?xf32, offset: 0, strides: [?, 1]>, %B: memref<?x?xf32, offset: 0, strides: [?, 1]>, %C: memref<?x?xf32, offset: 0, strides: [?, 1]>, %D: memref<?x?xf32, offset: 0, strides: [?, 1]>, %E: memref<?x?xf32, offset: 0, strides: [?, 1]>) -> memref<?x?xf32, offset: 0, strides: [?, 1]> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  %2 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, 1]>, memref<?x?xf32, offset: 0, strides: [?, 1]>, memref<?x?xf32, offset: 0, strides: [?, 1]>
  %c1 = constant 1 : index
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %C[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, 1]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, 1]>
}
// CHECK-LABEL: func @f1
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// No RAW dependences, the pass does not fuse RAR atm.
//      CHECK: linalg.matmul
//      CHECK: loop.for
//      CHECK:   loop.for
//      CHECK:     loop.for
//      CHECK:       linalg.matmul

func @f2(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f2
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//   CHECK-DAG:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//       CHECK:         linalg.matmul

func @f3(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %D, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f3
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f4(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %D) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f4
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// Fuse D then fuse C, no false dependence prevent it.
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f5(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %D, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%C, %B, %D) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %0 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %5 = std.subview %D[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %B[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f5
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//      CHECK-DAG:   %[[B_1:.*]] = dim %[[B]], 1 : memref<?x?xf32, #[[strided2D]]>
//      CHECK-DAG:   %[[D_0:.*]] = dim %[[D]], 0 : memref<?x?xf32, #[[strided2D]]>
//      CHECK-DAG:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
// Don't fuse C due to false dependence, note that this is too conservative though.
//          CHECK:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[B_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f6(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %E) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %0 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f6
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// Cannot fuse C due to interleaved read of C that would be bypassed.
// Cannot fuse E (WAW).
//   CHECK:   linalg.matmul
//   CHECK:   linalg.matmul
//   CHECK:   loop.for
//   CHECK:     loop.for
//   CHECK:       loop.for
//   CHECK:         linalg.matmul
// CHECK-NOT:       linalg.matmul

func @f7(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %C, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %3 = dim %C, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %4 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %E) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %7 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = std.subview %C[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%7, %9, %10) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  loop.for %arg5 = %c0 to %3 step %c2 {
    loop.for %arg6 = %c0 to %4 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %7 = std.subview %C[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %9 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%7, %9, %10) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f7
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//       CHECK:   %[[A_0:.*]] = dim %[[A]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[A_1:.*]] = dim %[[A]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[C_1:.*]] = dim %[[C]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[C_0:.*]] = dim %[[C]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   %[[D_1:.*]] = dim %[[D]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK:   linalg.matmul(%[[A]], %[[C]], %[[E]])
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[A_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[A_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//       CHECK:         linalg.matmul
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//   CHECK-NOT:         linalg.matmul

func @f8(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>, %E: memref<?x?xf32, offset: 0, strides: [?, ?]>) -> memref<?x?xf32, offset: 0, strides: [?, ?]> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = dim %A, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %A, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %C, %D) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %2 = dim %D, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = std.subview %A[%arg5, %arg7][%c2, %c4][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %6 = affine.apply #map2(%arg6)
        %7 = std.subview %D[%arg7, %arg6][%c4, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %8 = std.subview %E[%arg5, %arg6][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        linalg.matmul(%5, %7, %8) : memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
      }
    }
  }
  return %E : memref<?x?xf32, offset: 0, strides: [?, ?]>
}
// CHECK-LABEL: func @f8
//       CHECK:   (%[[A:.*]]: memref{{.*}}, %[[B:.*]]: memref{{.*}}, %[[C:.*]]: memref{{.*}}, %[[D:.*]]: memref{{.*}}, %[[E:.*]]: memref{{.*}})
//   CHECK:   linalg.matmul
//   CHECK:   linalg.matmul
//   CHECK:   loop.for
//   CHECK:     loop.for
//   CHECK:       loop.for
//   CHECK:         linalg.matmul
// CHECK-NOT:       linalg.matmul

#id_2d = (i, j) -> (i, j)
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  n_loop_types = [2, 0, 0],
  n_views = [2, 1]
}
func @pointwise(%A: memref<?x?xf32, offset: 0, strides: [?, ?]>, %B: memref<?x?xf32, offset: 0, strides: [?, ?]>, %C: memref<?x?xf32, offset: 0, strides: [?, ?]>, %D: memref<?x?xf32, offset: 0, strides: [?, ?]>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%E: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %E, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>, memref<?x?xf32, offset: 0, strides: [?, ?]>
  %0 = dim %B, 0 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  %1 = dim %B, 1 : memref<?x?xf32, offset: 0, strides: [?, ?]>
  loop.for %arg4 = %c0 to %0 step %c2 {
    loop.for %arg5 = %c0 to %1 step %c3 {
      %4 = std.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = std.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = std.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32, offset: 0, strides: [?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
    }
  }
  return
}
// CHECK-LABEL: func @pointwise
//       CHECK:   loop.for
//       CHECK:     loop.for
//   CHECK-NOT:   loop.for
//       CHECK:       linalg.generic
//       CHECK:         addf
//       CHECK:       linalg.generic
//       CHECK:         mulf

func @pointwise_no_view(%M: index, %N: index) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %A = alloc (%M, %N): memref<?x?xf32>
  %B = alloc (%M, %N): memref<?x?xf32>
  %C = alloc (%M, %N): memref<?x?xf32>
  %D = alloc (%M, %N): memref<?x?xf32>
  %E = alloc (%M, %N): memref<?x?xf32>
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%e: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %e, %arg5 : f32
    linalg.yield %2 : f32
  }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  %0 = dim %B, 0 : memref<?x?xf32>
  %1 = dim %B, 1 : memref<?x?xf32>
  loop.for %arg4 = %c0 to %0 step %c2 {
    loop.for %arg5 = %c0 to %1 step %c3 {
      %4 = std.subview %B[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      %5 = std.subview %C[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      %6 = std.subview %D[%arg4, %arg5][%c2, %c3][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>, memref<?x?xf32, offset: ?, strides: [?, ?]>
    }
  }
  return
}
// CHECK-LABEL: func @pointwise_no_view
//       CHECK:   loop.for
//       CHECK:     loop.for
//   CHECK-NOT:   loop.for
//       CHECK:       linalg.generic
//       CHECK:         addf
//       CHECK:       linalg.generic
//       CHECK:         mulf
