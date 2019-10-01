// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
//  TILE-02-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-002-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-234-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)

//   TILE-2-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
//  TILE-02-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// TILE-002-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// TILE-234-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
#strided1D = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
//   TILE-2-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
//  TILE-02-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// TILE-002-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// TILE-234-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
#strided2D = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)

func @matmul(%arg0: memref<?x?xf32, #strided2D>, %arg1: memref<?x?xf32, #strided2D>, %arg2: memref<?x?xf32, #strided2D>) {
  linalg.matmul(%arg0, %arg1, %arg2) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  return
}
// TILE-2-LABEL: func @matmul(
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}, %{{.*}}, %[[K]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[c]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   linalg.matmul(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>

// TILE-02-LABEL: func @matmul(
//       TILE-02: %[[N:.*]] = dim %arg1, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[K]], %{{.*}}, %{{.*}}, %[[b]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sCj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[c]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   linalg.matmul(%{{.*}}, %[[sBj]], %[[sCj]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>

// TILE-002-LABEL: func @matmul(
//       TILE-002: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002: loop.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-002:   %[[sAj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[a]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-002:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>

// TILE-234-LABEL: func @matmul(
//       TILE-234: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    loop.for %{{.*}} = %{{.*}}{{.*}} to %[[N]] step %{{.*}} {
//       TILE-234:      loop.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:        %[[ai:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:        %[[ak:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-234:        %[[sAik:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ai]], %{{.*}}, %{{.*}}, %[[ak]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:        %[[bk:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-234:        %[[bj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:        %[[sBkj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[bk]], %{{.*}}, %{{.*}}, %[[bj]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:        %[[ci:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:        %[[cj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:        %[[sCij:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ci]], %{{.*}}, %{{.*}}, %[[cj]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>

func @matvec(%arg0: memref<?x?xf32, #strided2D>, %arg1: memref<?xf32, #strided1D>, %arg2: memref<?xf32, #strided1D>) {
  linalg.matvec(%arg0, %arg1, %arg2) : memref<?x?xf32, #strided2D>, memref<?xf32, #strided1D>, memref<?xf32, #strided1D>
  return
}
// TILE-2-LABEL: func @matvec(
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[c]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-2:   linalg.matvec(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>

// TILE-02-LABEL: func @matvec(
//       TILE-02: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sAj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[a]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>

// TILE-002-LABEL: func @matvec(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @matvec(
//       TILE-234: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    loop.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:      %[[ai:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:      %[[aj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:      %[[sAij:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ai]], %{{.*}}, %{{.*}}, %[[aj]], %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:      %[[bj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:      %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[bj]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-234:      %[[ci:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:      %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ci]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//
//       TILE-234:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : memref<?x?xf32, #[[strided2D]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>

func @dot(%arg0: memref<?xf32, #strided1D>, %arg1: memref<?xf32, #strided1D>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, #strided1D>, memref<?xf32, #strided1D>, memref<f32>
  return
}
// TILE-2-LABEL: func @dot(
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-2: loop.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-2:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sBi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-2:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>, memref<f32>

// TILE-02-LABEL: func @dot(
//   TILE-02-NOT: loop.for

// TILE-002-LABEL: func @dot(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @dot(
//       TILE-234: %[[K:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-234:  loop.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:    %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:    %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-234:    %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:    %[[sBi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : memref<?xf32, #[[strided1D]]>
//       TILE-234:    linalg.dot(%[[sAi]], %[[sBi]], %{{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>, memref<f32>

func @fill(%arg0: memref<?x?xf32, #strided2D>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?xf32, #strided2D>, f32
  return
}
// TILE-2-LABEL: func @fill
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   fill{{.*}} f32

// TILE-02-LABEL: func @fill
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     fill{{.*}} f32

// TILE-002-LABEL: func @fill
//   TILE-002-NOT:   for
//       TILE-002:     fill{{.*}} f32

// TILE-234-LABEL: func @fill
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       fill{{.*}} f32

#id_2d = (i, j) -> (i, j)
#pointwise_2d_trait = {
  indexing_maps = [#id_2d, #id_2d, #id_2d],
  n_loop_types = [2, 0, 0],
  n_views = [2, 1]
}

func @pointwise(%arg0: memref<?x?xf32, #strided2D>, %arg1: memref<?x?xf32, #strided2D>,
                %arg2: memref<?x?xf32, #strided2D>) {
  linalg.generic #pointwise_2d_trait %arg0, %arg1, %arg2 {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %4 = addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }: memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  return
}
// TILE-2-LABEL: func @pointwise
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:   linalg.generic

// TILE-02-LABEL: func @pointwise
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:     linalg.generic

// TILE-002-LABEL: func @pointwise
//   TILE-002-NOT:   for
//       TILE-002:     linalg.generic

// TILE-234-LABEL: func @pointwise
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       linalg.generic
