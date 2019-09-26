// RUN: mlir-opt %s -linalg-fusion | FileCheck %s

#map0 = (d0) -> (d0 + 2)
#map1 = (d0) -> (d0 + 4)
#map2 = (d0) -> (d0 + 3)

func @f1(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = linalg.dim %A, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %A, 1 : !linalg.view<?x?xf32>
  %2 = linalg.dim %B, 1 : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %c1 = constant 1 : index
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %A[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %B[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %C[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f1
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
// No RAW dependences, the pass does not fuse RAR atm.
//      CHECK: linalg.matmul
//      CHECK: loop.for
//      CHECK:   loop.for
//      CHECK:     loop.for
//      CHECK:       linalg.matmul

func @f2(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %0 = linalg.dim %C, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %C, 1 : !linalg.view<?x?xf32>
  %2 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f2
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//   CHECK-DAG:   %[[C_0:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?x?xf32>
//   CHECK-DAG:   %[[C_1:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//   CHECK-DAG:   %[[D_1:.*]] = linalg.dim %[[D]], 1 : !linalg.view<?x?xf32>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       CHECK:         linalg.matmul
//       CHECK:         linalg.matmul

func @f3(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %0 = linalg.dim %D, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  %2 = linalg.dim %C, 1 : !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %D[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %C[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f3
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[D_0:.*]] = linalg.dim %[[D]], 0 : !linalg.view<?x?xf32>
//          CHECK:   %[[D_1:.*]] = linalg.dim %[[D]], 1 : !linalg.view<?x?xf32>
//          CHECK:   %[[C_1:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f4(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %0 = linalg.dim %C, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %C, 1 : !linalg.view<?x?xf32>
  %2 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f4
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//          CHECK:   %[[C_0:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?x?xf32>
//          CHECK:   %[[C_1:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//          CHECK:   %[[D_1:.*]] = linalg.dim %[[D]], 1 : !linalg.view<?x?xf32>
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
// Fuse D then fuse C, no false dependence prevent it.
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f5(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = linalg.dim %B, 1 : !linalg.view<?x?xf32>
  %1 = linalg.dim %D, 0 : !linalg.view<?x?xf32>
  %2 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %B, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %0 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %D[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %B[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f5
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//      CHECK-DAG:   %[[B_1:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//      CHECK-DAG:   %[[D_0:.*]] = linalg.dim %[[D]], 0 : !linalg.view<?x?xf32>
//      CHECK-DAG:   %[[D_1:.*]] = linalg.dim %[[D]], 1 : !linalg.view<?x?xf32>
// Don't fuse C due to false dependence, note that this is too conservative though.
//          CHECK:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//          CHECK:   loop.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//          CHECK:     loop.for %{{.*}} = %{{.*}} to %[[B_1]] step %{{.*}} {
//          CHECK:       loop.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//          CHECK:         linalg.matmul
//          CHECK:         linalg.matmul

func @f6(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = linalg.dim %C, 1 : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %C, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %1 = linalg.dim %C, 0 : !linalg.view<?x?xf32>
  %2 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %1 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %0 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %C[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
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

func @f7(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = linalg.dim %A, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %A, 1 : !linalg.view<?x?xf32>
  %2 = linalg.dim %C, 1 : !linalg.view<?x?xf32>
  %3 = linalg.dim %C, 0 : !linalg.view<?x?xf32>
  %4 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  linalg.matmul(%A, %C, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %5 = affine.apply #map0(%arg5)
        %6 = affine.apply #map1(%arg7)
        %7 = linalg.subview %A[%arg5, %5, %c1, %arg7, %6, %c1] : !linalg.view<?x?xf32>
        %8 = affine.apply #map2(%arg6)
        %9 = linalg.subview %C[%arg7, %6, %c1, %arg6, %8, %c1] : !linalg.view<?x?xf32>
        %10 = linalg.subview %E[%arg5, %5, %c1, %arg6, %8, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%7, %9, %10) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  loop.for %arg5 = %c0 to %3 step %c2 {
    loop.for %arg6 = %c0 to %4 step %c3 {
      loop.for %arg7 = %c0 to %2 step %c4 {
        %5 = affine.apply #map0(%arg5)
        %6 = affine.apply #map1(%arg7)
        %7 = linalg.subview %C[%arg5, %5, %c1, %arg7, %6, %c1] : !linalg.view<?x?xf32>
        %8 = affine.apply #map2(%arg6)
        %9 = linalg.subview %D[%arg7, %6, %c1, %arg6, %8, %c1] : !linalg.view<?x?xf32>
        %10 = linalg.subview %E[%arg5, %5, %c1, %arg6, %8, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%7, %9, %10) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f7
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
//       CHECK:   %[[A_0:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       CHECK:   %[[A_1:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       CHECK:   %[[C_1:.*]] = linalg.dim %[[C]], 1 : !linalg.view<?x?xf32>
//       CHECK:   %[[C_0:.*]] = linalg.dim %[[C]], 0 : !linalg.view<?x?xf32>
//       CHECK:   %[[D_1:.*]] = linalg.dim %[[D]], 1 : !linalg.view<?x?xf32>
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

func @f8(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %0 = linalg.dim %A, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %A, 1 : !linalg.view<?x?xf32>
  linalg.matmul(%A, %C, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %2 = linalg.dim %D, 1 : !linalg.view<?x?xf32>
  loop.for %arg5 = %c0 to %0 step %c2 {
    loop.for %arg6 = %c0 to %2 step %c3 {
      loop.for %arg7 = %c0 to %1 step %c4 {
        %3 = affine.apply #map0(%arg5)
        %4 = affine.apply #map1(%arg7)
        %5 = linalg.subview %A[%arg5, %3, %c1, %arg7, %4, %c1] : !linalg.view<?x?xf32>
        %6 = affine.apply #map2(%arg6)
        %7 = linalg.subview %D[%arg7, %4, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        %8 = linalg.subview %E[%arg5, %3, %c1, %arg6, %6, %c1] : !linalg.view<?x?xf32>
        linalg.matmul(%5, %7, %8) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      }
    }
  }
  return %E : !linalg.view<?x?xf32>
}
// CHECK-LABEL: func @f8
//       CHECK:   (%[[A:.*]]:{{.*}}, %[[B:.*]]:{{.*}}, %[[C:.*]]:{{.*}}, %[[D:.*]]:{{.*}}, %[[E:.*]]:{{.*}})
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
func @pointwise(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  linalg.generic #pointwise_2d_trait %A, %A, %B {
  ^bb0(%E: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %2 = addf %E, %arg5 : f32
    linalg.yield %2 : f32
  }: !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  %0 = linalg.dim %B, 0 : !linalg.view<?x?xf32>
  %1 = linalg.dim %B, 1 : !linalg.view<?x?xf32>
  loop.for %E = %c0 to %0 step %c2 {
    loop.for %arg5 = %c0 to %1 step %c3 {
      %2 = affine.apply #map0(%E)
      %3 = affine.apply #map1(%arg5)
      %4 = linalg.subview %B[%E, %2, %c1, %arg5, %3, %c1] : !linalg.view<?x?xf32>
      %5 = linalg.subview %C[%E, %2, %c1, %arg5, %3, %c1] : !linalg.view<?x?xf32>
      %6 = linalg.subview %D[%E, %2, %c1, %arg5, %3, %c1] : !linalg.view<?x?xf32>
      linalg.generic #pointwise_2d_trait %4, %5, %6 {
      ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):       // no predecessors
        %7 = mulf %arg6, %arg7 : f32
        linalg.yield %7 : f32
      }: !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
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
