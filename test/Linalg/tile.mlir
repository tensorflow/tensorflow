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

func @matmul(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?x?xf32>, %arg2: !linalg.view<?x?xf32>) {
  linalg.matmul(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// TILE-2-LABEL: func @matmul(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[K:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}, %{{.*}}, %[[K]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[N:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[c]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-2:   linalg.matmul(%[[sAi]], %{{.*}}, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-02-LABEL: func @matmul(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>) {
//       TILE-02: %[[N:.*]] = linalg.dim %arg1, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       TILE-02:   %[[K:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[K]], %{{.*}}, %{{.*}}, %[[b]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-02:   %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sCj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[c]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-02:   linalg.matmul(%{{.*}}, %[[sBj]], %[[sCj]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-002-LABEL: func @matmul(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>) {
//       TILE-002: %[[K:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-002: linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-002:   %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-002:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-002:   %[[sAj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[a]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-002:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-002:   %[[N:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-002:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-002:   linalg.matmul(%[[sAj]], %[[sBj]], %{{.*}}) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-234-LABEL: func @matmul(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?x?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-234: %[[N:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[N]] step %{{.*}} {
//       TILE-234:      linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:        %[[ai:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:        %[[ak:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-234:        %[[sAik:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ai]], %{{.*}}, %{{.*}}, %[[ak]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-234:        %[[bk:.*]] = affine.apply #[[UB2]](%{{.*}})
//       TILE-234:        %[[bj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:        %[[sBkj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[bk]], %{{.*}}, %{{.*}}, %[[bj]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-234:        %[[ci:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:        %[[cj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:        %[[sCij:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ci]], %{{.*}}, %{{.*}}, %[[cj]], %{{.*}}] : !linalg.view<?x?xf32>
//
//       TILE-234:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>) {
  linalg.matvec(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// TILE-2-LABEL: func @matvec(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-2: linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[N:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}, %{{.*}}, %[[N]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-2:   %[[c:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[c]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-2:   linalg.matvec(%[[sAi]], %{{.*}}, %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-02-LABEL: func @matvec(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//       TILE-02: %[[K:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-02: linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-02:   %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-02:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sAj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[M]], %{{.*}}, %{{.*}}, %[[a]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-02:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-02:   %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-02:   linalg.matvec(%[[sAj]], %[[sBj]], %{{.*}}) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-002-LABEL: func @matvec(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @matvec(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//       TILE-234: %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?x?xf32>
//       TILE-234: %[[K:.*]] = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//       TILE-234:  linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-234:    linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:      %[[ai:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:      %[[aj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:      %[[sAij:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ai]], %{{.*}}, %{{.*}}, %[[aj]], %{{.*}}] : !linalg.view<?x?xf32>
//       TILE-234:      %[[bj:.*]] = affine.apply #[[UB1]](%{{.*}})
//       TILE-234:      %[[sBj:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[bj]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-234:      %[[ci:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:      %[[sCi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[ci]], %{{.*}}] : !linalg.view<?xf32>
//
//       TILE-234:      linalg.matvec(%[[sAij]], %[[sBj]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

func @dot(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// TILE-2-LABEL: func @dot(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//       TILE-2: %[[M:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?xf32>
//       TILE-2: linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[M]] step %{{.*}} {
//       TILE-2:   %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-2:   %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-2:   %[[sBi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-2:   linalg.dot(%[[sAi]], %[[sBi]], {{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

// TILE-02-LABEL: func @dot(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//   TILE-02-NOT: linalg.for

// TILE-002-LABEL: func @dot(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//   TILE-002-NOT: linalg.for

// TILE-234-LABEL: func @dot(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//       TILE-234: %[[K:.*]] = linalg.dim %{{.*}}, 0 : !linalg.view<?xf32>
//       TILE-234:  linalg.for %{{.*}} = %{{.*}}{{.*}} to %[[K]] step %{{.*}} {
//       TILE-234:    %[[a:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:    %[[sAi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[a]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-234:    %[[b:.*]] = affine.apply #[[UB0]](%{{.*}})
//       TILE-234:    %[[sBi:.*]] = linalg.subview %{{.*}}[%{{.*}}, %[[b]], %{{.*}}] : !linalg.view<?xf32>
//       TILE-234:    linalg.dot(%[[sAi]], %[[sBi]], %{{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

func @fill(%arg0: !linalg.view<?x?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?x?xf32>, f32
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
