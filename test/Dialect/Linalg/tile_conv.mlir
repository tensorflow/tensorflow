// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,0,0,4 | FileCheck %s -check-prefix=TILE-23004

// TILE-23004-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-23004-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-23004-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)
// TILE-23004-DAG: #[[D0x30pS0x10:.*]] = (d0) -> (d0 * 30)
// TILE-23004-DAG: #[[D0x30pS0x10p90:.*]] = (d0)[s0] -> (d0 * 30 + s0 * 10 + 90)
// TILE-23004-DAG: #[[strided4D:.*]] = (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)

func @conv(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// TILE-23004-LABEL: func @conv(
//       TILE-23004:   %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       TILE-23004:   %[[Q:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[B:.*]] = dim %{{.*}}, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[PaddedInput0:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   %[[X0:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       TILE-23004:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       TILE-23004:       loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       TILE-23004:       %[[Z0:.*]] = dim %{{.*}}, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[Z1:.*]] = dim %{{.*}}, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[I2p4:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-23004:         %[[K:.*]] = dim %{{.*}}, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[FilterView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[Z0]], %{{.*}}, %{{.*}}, %[[Z1]], %{{.*}}, %{{.*}}, %[[I2p4]], %{{.*}}, %{{.*}}, %[[K]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//
//       TILE-23004:         %[[I0p3:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-23004:         %[[I1:.*]] = affine.apply #[[D0x30pS0x10]](%{{.*}})
//       TILE-23004:         %[[I1pStep:.*]] = affine.apply #[[D0x30pS0x10p90]](%{{.*}})[%[[PaddedInput0]]]
//       TILE-23004:         %[[SZ2:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[I2p2:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-23004:         %[[InputView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[I0p3]], %{{.*}}, %[[I1]], %[[I1pStep]], %{{.*}}, %{{.*}}, %[[SZ2]], %{{.*}}, %{{.*}}, %[[I2p2]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//
//       TILE-23004:         %[[B:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-23004:         %[[I1p3:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-23004:         %[[X0:.*]] = dim %{{.*}}, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[X1:.*]] = dim %{{.*}}, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       TILE-23004:         %[[OutputView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[B]], %{{.*}}, %{{.*}}, %[[I1p3]], %{{.*}}, %{{.*}}, %[[X0]], %{{.*}}, %{{.*}}, %[[X1]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//
//       TILE-23004:         linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) {dilations = [10, 20], strides = [30, 40]} : memref<?x?x?x?xf32, #[[strided4D]]>, memref<?x?x?x?xf32, #[[strided4D]]>, memref<?x?x?x?xf32, #[[strided4D]]>
