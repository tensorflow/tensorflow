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
// TILE-23004-LABEL: func @conv(%arg0: !linalg.view<?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?xf32>) {
//       TILE-23004:  %[[Q:.*]] = linalg.dim %arg0, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:  %[[B:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?x?x?xf32>
//       TILE-23004: %[[PaddedInput0:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004: %[[X0:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004: linalg.for %i0 = %c0 to %[[B]] step %c2 {
//       TILE-23004:   linalg.for %i1 = %c0 to %[[X0]] step %c3 {
//       TILE-23004:     linalg.for %i2 = %c0 to %[[Q]] step %c4 {
//       TILE-23004:       %[[Z0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[Z1:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[I2p4:.*]] = affine.apply #[[UB2]](%i2)
//       TILE-23004:       %[[K:.*]] = linalg.dim %arg0, 3 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[FilterView:.*]] = linalg.subview %arg0[%c0, %[[Z0]], %c1, %c0, %[[Z1]], %c1, %i2, %[[I2p4]], %c1, %c0, %[[K]], %c1] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:       %[[I0p3:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-23004:       %[[I1:.*]] = affine.apply #[[D0x30pS0x10]](%i1)[%c0]
//       TILE-23004:       %[[I1pStep:.*]] = affine.apply #[[D0x30pS0x10p90]](%i1)[%[[PaddedInput0]]]
//       TILE-23004:       %[[SZ2:.*]] = linalg.dim %arg1, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[I2p2:.*]] = affine.apply #[[UB2]](%i2)
//       TILE-23004:       %[[InputView:.*]] = linalg.subview %arg1[%i0, %[[I0p3]], %c1, %[[I1]], %[[I1pStep]], %c1, %c0, %[[SZ2]], %c1, %i2, %[[I2p2]], %c1] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:       %[[B:.*]] = affine.apply #[[UB0]](%i0)
//       TILE-23004:       %[[I1p3:.*]] = affine.apply #[[UB1]](%i1)
//       TILE-23004:       %[[X0:.*]] = linalg.dim %arg2, 2 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[X1:.*]] = linalg.dim %arg2, 3 : !linalg.view<?x?x?x?xf32>
//       TILE-23004:       %[[OutputView:.*]] = linalg.subview %arg2[%i0, %[[B]], %c1, %i1, %[[I1p3]], %c1, %c0, %[[X0]], %c1, %c0, %[[X1]], %c1] : !linalg.view<?x?x?x?xf32>
//
//       TILE-23004:       linalg.conv(%[[FilterView]], %[[InputView]], %[[OutputView]]) : !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>
