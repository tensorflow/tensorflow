// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,0,0,4 | FileCheck %s -check-prefix=TILE-23004

// TILE-23004-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-23004-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-23004-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)
// TILE-23004-DAG: #[[D0x30pS0x10:.*]] = (d0)[s0] -> (d0 * 30 + s0 * 10)
// TILE-23004-DAG: #[[D0x30pS0x10p90:.*]] = (d0)[s0] -> (d0 * 30 + s0 * 10 + 90)
func @conv(%arg0: !linalg.view<?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [10, 20], strides = [30, 40]} : !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>
  return
}
// TILE-23004-LABEL: func @conv(%{{.*}}: !linalg.view<?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?xf32>) {
//       TILE-23004:   %[[Q:.*]] = linalg.dim %{{.*}}, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:   %[[B:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:   %[[PaddedInput0:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:   %[[X0:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       TILE-23004:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       TILE-23004:       loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       TILE-23004:       %[[Z0:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[Z1:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[I2p4:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-23004:         %[[K:.*]] = linalg.dim %{{.*}}, 3 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[FilterView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[Z0]], %{{.*}}, %{{.*}}, %[[Z1]], %{{.*}}, %{{.*}}, %[[I2p4]], %{{.*}}, %{{.*}}, %[[K]], %{{.*}}] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:         %[[I0p3:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-23004:         %[[I1:.*]] = affine.apply #[[D0x30pS0x10]](%{{.*}})[%{{.*}}]
//       TILE-23004:         %[[I1pStep:.*]] = affine.apply #[[D0x30pS0x10p90]](%{{.*}})[%[[PaddedInput0]]]
//       TILE-23004:         %[[SZ2:.*]] = linalg.dim %{{.*}}, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[I2p2:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-23004:         %[[InputView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[I0p3]], %{{.*}}, %[[I1]], %[[I1pStep]], %{{.*}}, %{{.*}}, %[[SZ2]], %{{.*}}, %{{.*}}, %[[I2p2]], %{{.*}}] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:         %[[B:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-23004:         %[[I1p3:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-23004:         %[[X0:.*]] = linalg.dim %{{.*}}, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[X1:.*]] = linalg.dim %{{.*}}, 3 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:         %[[OutputView:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[B]], %{{.*}}, %{{.*}}, %[[I1p3]], %{{.*}}, %{{.*}}, %[[X0]], %{{.*}}, %{{.*}}, %[[X1]], %{{.*}}] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:         linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) {dilations = [10, 20], strides = [30, 40]} : !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>
