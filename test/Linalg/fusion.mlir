// RUN: mlir-opt %s -linalg-fusion -linalg-fusion-tile-sizes=0,0,0 | FileCheck %s -check-prefix=FUSE-0
// RUN: mlir-opt %s -linalg-fusion -linalg-fusion-tile-sizes=2 | FileCheck %s -check-prefix=FUSE-2
// RUN: mlir-opt %s -linalg-fusion -linalg-fusion-tile-sizes=2,3 | FileCheck %s -check-prefix=FUSE-23
// RUN: mlir-opt %s -linalg-fusion -linalg-fusion-tile-sizes=2,3,4 | FileCheck %s -check-prefix=FUSE-234

func @f1(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// No RAW dependences, the pass does not fuse RAR atm.
// FUSE-0-LABEL: func @f1
//   FUSE-0-NOT: linalg.for
// FUSE-2-LABEL: func @f1
//   FUSE-2-NOT: linalg.for
// FUSE-23-LABEL: func @f1
//   FUSE-23-NOT: linalg.for
// FUSE-234-LABEL: func @f1
//   FUSE-234-NOT: linalg.for

func @f2(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// No tiling => no fusion
// FUSE-0-LABEL: func @f2
//   FUSE-0-NOT: linalg.for
//
// FUSE-2-LABEL: func @f2
//       FUSE-2:   %[[C_0:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       FUSE-2:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-2:     linalg.matmul
//       FUSE-2:     linalg.matmul
//
// FUSE-23-LABEL: func @f2
//       FUSE-23:   %[[C_0:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//       FUSE-23:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-23:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-23:     linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//
// FUSE-234-LABEL: func @f2
//       FUSE-234:   %[[C_0:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-234:     linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-234:       linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul

func @f3(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%D, %C, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// No tiling => no fusion
// FUSE-0-LABEL: func @f3
//   FUSE-0-NOT: linalg.for
//
// Read to %C does not get tiled along 1st dimension => no fusion
// FUSE-2-LABEL: func @f3
//   FUSE-2-NOT:   linalg.for
//
// FUSE-23-LABEL: func @f3
//       FUSE-23:   %[[D_0:.*]] = linalg.dim %arg3, 0 : !linalg.view<?x?xf32>
//       FUSE-23:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-23:   linalg.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//       FUSE-23:     linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//
// FUSE-234-LABEL: func @f3
//       FUSE-234:   %[[D_0:.*]] = linalg.dim %arg3, 0 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   linalg.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//       FUSE-234:     linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-234:       linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul

func @f4(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// No tiling => no fusion
// FUSE-0-LABEL: func @f4
//   FUSE-0-NOT: linalg.for
//
// Read to %D does not get tiled along 1st dimension => no fusion
// FUSE-2-LABEL: func @f4
//       FUSE-2:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-2:   %[[C_0:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       FUSE-2:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-2:     linalg.matmul
//       FUSE-2:     linalg.matmul
//
// FUSE-23-LABEL: func @f4
//       FUSE-23:   %[[C_0:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//       FUSE-23:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-23:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-23:     linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//
// FUSE-234-LABEL: func @f4
//       FUSE-234:   %[[C_0:.*]] = linalg.dim %arg2, 0 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   linalg.for %{{.*}} = %{{.*}} to %[[C_0]] step %{{.*}} {
//       FUSE-234:     linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-234:       linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul

func @f5(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %B, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%D, %B, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// No tiling => no fusion
// FUSE-0-LABEL: func @f5
//   FUSE-0-NOT: linalg.for
//
// FUSE-2-LABEL: func @f5
//       FUSE-2:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-2:   %[[D_0:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       FUSE-2:   linalg.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//       FUSE-2:     linalg.matmul
//       FUSE-2:     linalg.matmul
//
// FUSE-23-LABEL: func @f5
//       FUSE-23:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-23:   %[[D_0:.*]] = linalg.dim %arg3, 0 : !linalg.view<?x?xf32>
//       FUSE-23:   %[[B_1:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       FUSE-23:   linalg.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//       FUSE-23:     linalg.for %{{.*}} = %{{.*}} to %[[B_1]] step %{{.*}} {
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//
// FUSE-234-LABEL: func @f5
//       FUSE-234:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-234:   %[[D_0:.*]] = linalg.dim %arg3, 0 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[D_1:.*]] = linalg.dim %arg3, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[B_1:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   linalg.for %{{.*}} = %{{.*}} to %[[D_0]] step %{{.*}} {
//       FUSE-234:     linalg.for %{{.*}} = %{{.*}} to %[[B_1]] step %{{.*}} {
//       FUSE-234:       linalg.for %{{.*}} = %{{.*}} to %[[D_1]] step %{{.*}} {
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul

func @f6(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %C, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// Write to %C can not be fused because the 2 RAW are not compatible.
// The current algorithm just bails out on fusion in the case of any write-based
// interleaved dependence.
// No tiling => no fusion
// FUSE-0-LABEL: func @f6
//   FUSE-0-NOT: linalg.for
//
// Read to D is not tiled along 1st dimension => no fusion
// FUSE-2-LABEL: func @f6
//   FUSE-2-NOT:   linalg.for
//
// FUSE-23-LABEL: func @f6
//
// FUSE-234-LABEL: func @f6

func @f7(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %C, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %C, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%C, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// The only fusion that respects dependences is the write to %C into the
// immediately following read.
// No tiling => no fusion
// FUSE-0-LABEL: func @f7
//   FUSE-0-NOT: linalg.for
//
// Read to %C (in 3rd matmul) is not tiled along 1st dimension => no fusion
// FUSE-2-LABEL: func @f7
//   FUSE-2-NOT:   linalg.for
//
// FUSE-23-LABEL: func @f7
//       FUSE-23:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-23:   %[[A_0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       FUSE-23:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-23:   linalg.for %{{.*}} = %{{.*}} to %[[A_0]] step %{{.*}} {
//       FUSE-23:     linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-23:       linalg.matmul
//       FUSE-23:       linalg.matmul
//       FUSE-23:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//
// FUSE-234-LABEL: func @f7
//       FUSE-234:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})
//       FUSE-234:   %[[A_0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[A_1:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   %[[C_1:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?xf32>
//       FUSE-234:   linalg.for %{{.*}} = %{{.*}} to %[[A_0]] step %{{.*}} {
//       FUSE-234:     linalg.for %{{.*}} = %{{.*}} to %[[C_1]] step %{{.*}} {
//       FUSE-234:       linalg.for %{{.*}} = %{{.*}} to %[[A_1]] step %{{.*}} {
//       FUSE-234:         linalg.matmul
//       FUSE-234:         linalg.matmul
//       FUSE-234:   linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}})

func @f8(%A: !linalg.view<?x?xf32>, %B: !linalg.view<?x?xf32>, %C: !linalg.view<?x?xf32>, %D: !linalg.view<?x?xf32>, %E: !linalg.view<?x?xf32>) -> !linalg.view<?x?xf32> {
  linalg.matmul(%A, %C, %D) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matmul(%A, %D, %E) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return %E : !linalg.view<?x?xf32>
}
// In this example, %D can never be fused because the WAR on %C would be violated
// No tiling => no fusion
// FUSE-0-LABEL: func @f8
//   FUSE-0-NOT: linalg.for
//
// FUSE-2-LABEL: func @f8
//   FUSE-2-NOT:   linalg.for
//
// FUSE-23-LABEL: func @f8
//   FUSE-23-NOT:   linalg.for
//
// FUSE-234-LABEL: func @f8
//   FUSE-234-NOT:   linalg.for
