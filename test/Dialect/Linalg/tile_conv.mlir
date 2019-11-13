// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,0,0,4 | FileCheck %s -check-prefix=TILE-23004

// TILE-23004-DAG: #[[D0x30pS0x10:.*]] = (d0) -> (d0 * 30)
// TILE-23004-DAG: #[[S0x10p90:.*]] = ()[s0] -> (s0 * 10 + 90)
// TILE-23004-DAG: #[[strided4D:.*]] = (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)
// TILE-23004-DAG: #[[strided4D_dynamic:.*]] = (d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)

func @conv(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// TILE-23004-LABEL: func @conv(
//       TILE-23004:   %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       TILE-23004-DAG: %[[C0:.*]] = constant 0 : index
//       TILE-23004-DAG: %[[C1:.*]] = constant 1 : index
//       TILE-23004-DAG: %[[C2:.*]] = constant 2 : index
//       TILE-23004-DAG: %[[C3:.*]] = constant 3 : index
//       TILE-23004-DAG: %[[C4:.*]] = constant 4 : index
//       TILE-23004:   %[[Q:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[B:.*]] = dim %{{.*}}, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[PaddedInput0:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[X0:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   loop.for %[[ivI:.*]] = %{{.*}} to %[[B]] step %{{.*}} {
//       TILE-23004:     loop.for %[[ivJ:.*]] = %{{.*}} to %[[X0]] step %{{.*}} {
//       TILE-23004:       loop.for %[[ivK:.*]] = %{{.*}} to %[[Q]] step %{{.*}} {
//       TILE-23004:         %[[Z0:.*]] = dim %{{.*}}, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[Z1:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[K:.*]] = dim %{{.*}}, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[FilterView:.*]] = std.subview %{{.*}}[%[[C0]], %[[C0]], %[[ivK]], %[[C0]]][%[[Z0]], %[[Z1]], %[[C4]], %[[K]]][%[[C1]], %[[C1]], %[[C1]], %[[C1]]] : memref<?x?x?x?xf32, #[[strided4D]]> to memref<?x?x?x?xf32, #[[strided4D_dynamic]]>
//
//       TILE-23004:         %[[J1:.*]] = affine.apply #[[D0x30pS0x10]](%[[ivJ]])
//       T__ILE-23004:         %[[I1pStep:.*]] = affine.apply #[[S0x10p90]]()[%[[I1]]]
//       TILE-23004:         %[[SZ2:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[InputView:.*]] = std.subview %{{.*}}[%[[ivI]], %[[J1]], %[[C0]], %[[ivK]]][%[[C2]], %{{.*}}, %[[SZ2]], %[[C4]]][%[[C1]], %[[C1]], %[[C1]], %[[C1]]] : memref<?x?x?x?xf32, #[[strided4D]]> to memref<?x?x?x?xf32, #[[strided4D_dynamic]]>
//
//       TILE-23004:         %[[X0:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[X1:.*]] = dim %{{.*}}, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[OutputView:.*]] = std.subview %{{.*}}[%[[ivI]], %[[ivJ]], %[[C0]], %[[C0]]][%[[C2]], %[[C3]], %[[X0]], %[[X1]]][%[[C1]], %[[C1]], %[[C1]], %[[C1]]] : memref<?x?x?x?xf32, #[[strided4D]]> to memref<?x?x?x?xf32, #[[strided4D_dynamic]]>
//
//       TILE-23004:         linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, #[[strided4D_dynamic]]>, memref<?x?x?x?xf32, #[[strided4D_dynamic]]>, memref<?x?x?x?xf32, #[[strided4D_dynamic]]>
