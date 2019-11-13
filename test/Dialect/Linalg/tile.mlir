// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
//  TILE-02-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// TILE-002-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// TILE-234-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)

//   TILE-2-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
//  TILE-02-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// TILE-002-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// TILE-234-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)

//   TILE-2-DAG: #[[strided1D_dynamic:.*]] = (d0)[s0, s1] -> (d0 * s1 + s0)
//   TILE-02-DAG: #[[strided1D_dynamic:.*]] = (d0)[s0, s1] -> (d0 * s1 + s0)
//   T_ILE-002-DAG: #[[strided1D_dynamic:.*]] = (d0)[s0, s1] -> (d0 * s1 + s0)
//   TILE-234-DAG: #[[strided1D_dynamic:.*]] = (d0)[s0, s1] -> (d0 * s1 + s0)

//   TILE-2-DAG: #[[strided2D_dynamic:.*]] = (d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)
//   TILE-02-DAG: #[[strided2D_dynamic:.*]] = (d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)
//   TILE-002-DAG: #[[strided2D_dynamic:.*]] = (d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)
//   TILE-234-DAG: #[[strided2D_dynamic:.*]] = (d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)

//   REACTIVATE_ME_TILE-2-DAG: #[[stride_99_1_layout_map:.*]] = (d0, d1)[s0] -> (d0 * 99 + s0 + d1)
//  REACTIVATE_ME_TILE-02-DAG: #[[stride_99_1_layout_map:.*]] = (d0, d1)[s0] -> (d0 * 99 + s0 + d1)
// REACTIVATE_ME_TILE-234-DAG: #[[stride_99_1_layout_map:.*]] = (d0, d1)[s0] -> (d0 * 99 + s0 + d1)

func @matmul(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%arg0, %arg1, %arg2) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// TILE-2-LABEL: func @matmul(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = std.subview %{{.*}}[%[[I]], %[[C0]]][%[[C2]], %[[K]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sCi:.*]] = std.subview %{{.*}}[%[[I]], %[[C0]]][%[[C2]], %[[N]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   linalg.matmul(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// TILE-02-LABEL: func @matmul(
//       TILE-02-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-02-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-02-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-02: %[[N:.*]] = dim %arg1, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %[[J:.*]] = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[sBj:.*]] = std.subview %{{.*}}[%[[C0]], %[[J]]][%[[K]], %[[C2]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[sCj:.*]] = std.subview %{{.*}}[%[[C0]], %[[J]]][%[[M]], %[[C2]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   linalg.matmul(%{{.*}}, %[[sBj]], %[[sCj]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

// TILE-002-LABEL: func @matmul(
//       TILE-002-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-002-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-002-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-002: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002: loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[sAj:.*]] = std.subview %{{.*}}[%[[C0]], %[[K]]][%[[M]], %[[C2]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-002:   %[[sBj:.*]] = std.subview %{{.*}}[%[[K]], %[[C0]]][%[[C2]], %[[N]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D]]>

// TILE-234-LABEL: func @matmul(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-234-DAG: %[[C4:.*]] = constant 4 : index
//       TILE-234: %[[ubM:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[ubK:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[ubN:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[ubM]] step %{{.*}} {
//       TILE-234:    loop.for %[[J:.*]] = %{{.*}}{{.*}} to %[[ubN]] step %{{.*}} {
//       TILE-234:      loop.for %[[K:.*]] = %{{.*}}{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:        %[[sAik:.*]] = std.subview %{{.*}}[%[[I]], %[[K]]][%[[C2]], %[[C4]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:        %[[sBkj:.*]] = std.subview %{{.*}}[%[[K]], %[[J]]][%[[C4]], %[[C3]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:        %[[sCij:.*]] = std.subview %{{.*}}[%[[I]], %[[J]]][%[[C2]], %[[C3]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D_dynamic]]>

func @matvec(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec(%arg0, %arg1, %arg2) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// TILE-2-LABEL: func @matvec(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2: loop.for %[[I]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[N:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-2:   %[[sAi:.*]] = std.subview %{{.*}}[%[[I]], %[[C0]]][%[[C2]], %[[N]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-2:   %[[sCi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   linalg.matvec(%[[sAi]], %{{.*}}, %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D_dynamic]]>

// TILE-02-LABEL: func @matvec(
//       TILE-02-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-02-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-02-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-02: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02: loop.for %[[J]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-02:   %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-02:   %[[sAj:.*]] = std.subview %{{.*}}[%[[C0]], %[[J]]][%[[M]], %[[C2]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-02:   %[[sBj:.*]] = std.subview %{{.*}}[%[[J]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %{{.*}}) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D]]>

// TILE-002-LABEL: func @matvec(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @matvec(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-234: %[[M:.*]] = dim %{{.*}}, 0 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234: %[[K:.*]] = dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    loop.for %[[J:.*]] = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:      %[[sAij:.*]] = std.subview %{{.*}}[%[[I]], %[[J]]][%[[C2]], %[[C3]]][%[[C1]], %[[C1]]] : memref<?x?xf32, #[[strided2D]]> to memref<?x?xf32, #[[strided2D_dynamic]]>
//       TILE-234:      %[[sBj:.*]] = std.subview %{{.*}}[%[[J]]][%[[C3]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:      %[[sCi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//
//       TILE-234:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>

func @dot(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// TILE-2-LABEL: func @dot(
//       TILE-2-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-2-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-2-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-2: %[[M:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-2: loop.for %[[I]] = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[sAi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   %[[sBi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-2:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<f32>

// TILE-02-LABEL: func @dot(
//   TILE-02-NOT: loop.for

// TILE-002-LABEL: func @dot(
//   TILE-002-NOT: loop.for

// TILE-234-LABEL: func @dot(
//       TILE-234-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-234-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-234-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-234:  %[[ubK:.*]] = dim %{{.*}}, 0 : memref<?xf32, #[[strided1D]]>
//       TILE-234:  loop.for %[[I:.*]] = %{{.*}} to %[[ubK]] step %{{.*}} {
//       TILE-234:    %[[sAi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:    %[[sBi:.*]] = std.subview %{{.*}}[%[[I]]][%[[C2]]][%[[C1]]] : memref<?xf32, #[[strided1D]]> to memref<?xf32, #[[strided1D_dynamic]]>
//       TILE-234:    linalg.dot(%[[sAi]], %[[sBi]], %{{.*}}) : memref<?xf32, #[[strided1D_dynamic]]>, memref<?xf32, #[[strided1D_dynamic]]>, memref<f32>

func @fill_static(%arg0: memref<127x99xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<127x99xf32>, f32
  return
}
// TILE-2-LABEL: func @fill_static
//       TILE-2:   for
//   TILE-2-NOT:   for
//       TILE-2:       std.subview{{.*}} : memref<127x99xf32>
//       TILE-2:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32

// TILE-02-LABEL: func @fill_static
//       TILE-02:   for
//   TILE-02-NOT:   for
//       TILE-02:       std.subview{{.*}} : memref<127x99xf32>
//       TILE-02:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32

// TILE-002-LABEL: func @fill_static
//   TILE-002-NOT:   for
//       TILE-002:     linalg.fill{{.*}} memref<127x99xf32>, f32

// TILE-234-LABEL: func @fill_static
//       TILE-234:   for
//       TILE-234:     for
//   TILE-234-NOT:   for
//       TILE-234:       std.subview{{.*}} : memref<127x99xf32>
//       TILE-234:       linalg.fill{{.*}} : memref<?x?xf32, #[[strided2D_dynamic]]>, f32


func @fill(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?xf32, offset: ?, strides: [?, 1]>, f32
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

func @pointwise(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                %arg2: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.generic #pointwise_2d_trait %arg0, %arg1, %arg2 {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):   // no predecessors
    %4 = addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }: memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
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
