// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2 | FileCheck %s -check-prefix=TILE-2
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,2 | FileCheck %s -check-prefix=TILE-02
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=0,0,2 | FileCheck %s -check-prefix=TILE-002
// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 | FileCheck %s -check-prefix=TILE-234

//   TILE-2-DAG: #[[ID:.*]] = (d0) -> (d0)
//   TILE-2-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
//  TILE-02-DAG: #[[ID:.*]] = (d0) -> (d0)
//  TILE-02-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-002-DAG: #[[ID:.*]] = (d0) -> (d0)
// TILE-002-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[ID:.*]] = (d0) -> (d0)
// TILE-234-DAG: #[[UB0:.*]] = (d0) -> (d0 + 2)
// TILE-234-DAG: #[[UB1:.*]] = (d0) -> (d0 + 3)
// TILE-234-DAG: #[[UB2:.*]] = (d0) -> (d0 + 4)

func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %K = linalg.range %c0:%arg3:%c1 : !linalg.range
  %A = linalg.view %arg0[%I, %K] : !linalg.view<?x?xf32>
  %B = linalg.view %arg0[%K, %J] : !linalg.view<?x?xf32>
  %C = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// TILE-2-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-2: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-2: affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.slice %[[A]][%[[ra]], %2] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[c]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sCi:.*]] = linalg.slice %[[C]][%[[rc]], %1] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-2-NEXT:   linalg.matmul(%[[sAi]], %[[B]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-02-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-02: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-02: affine.for %i0 = #[[ID]](%c0_0) to #[[ID]](%arg2) step 2 {
//  TILE-02-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[b]]:%c2 : !linalg.range
//  TILE-02-NEXT:   %[[sBj:.*]] = linalg.slice %[[B]][%{{.*}}, %[[rb]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[c]]:%c2 : !linalg.range
//  TILE-02-NEXT:   %[[sCj:.*]] = linalg.slice %[[C]][%{{.*}}, %[[rc]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-02-NEXT:   linalg.matmul(%[[A]], %[[sBj]], %[[sCj]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-002-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-002: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-002-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-002-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-002: affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg3) step 2 {
//  TILE-002-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2 : !linalg.range
//  TILE-002-NEXT:   %[[sAj:.*]] = linalg.slice %[[A]][%{{.*}}, %[[ra]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-002-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-002-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[b]]:%c2 : !linalg.range
//  TILE-002-NEXT:   %[[sBj:.*]] = linalg.slice %[[B]][%[[rb]], %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-002-NEXT:   linalg.matmul(%[[sAj]], %[[sBj]], %[[C]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

// TILE-234-LABEL: func @matmul(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-234: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       TILE-234:  affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-234-NEXT:    affine.for %i1 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg2) step 3 {
//  TILE-234-NEXT:      affine.for %i2 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg3) step 4 {
//  TILE-234-NEXT:        %[[ai:.*]]  = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[rai:.*]] = linalg.range %i0:%[[ai]]:%c2{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[ak:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[rak:.*]] = linalg.range %i2:%[[ak]]:%c4{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[sAik:.*]] = linalg.slice %[[A]][%[[rai]], %[[rak]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[bk:.*]] = affine.apply #[[UB2]](%i2)
//  TILE-234-NEXT:        %[[rbk:.*]] = linalg.range %i2:%[[bk]]:%c4{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[bj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[rbj:.*]] = linalg.range %i1:%[[bj]]:%c3{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[sBkj:.*]] = linalg.slice %[[B]][%[[rbk]], %[[rbj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-234-NEXT:        %[[ci:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:        %[[rci:.*]] = linalg.range %i0:%[[ci]]:%c2{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[cj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:        %[[rcj:.*]] = linalg.range %i1:%[[cj]]:%c3{{.*}} : !linalg.range
//  TILE-234-NEXT:        %[[sCij:.*]] = linalg.slice %[[C]][%[[rci]], %[[rcj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-234-NEXT:        linalg.matmul(%[[sAik]], %[[sBkj]], %[[sCij]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %2 = linalg.view %arg0[%I, %J] : !linalg.view<?x?xf32>
  %3 = linalg.view %arg0[%J] : !linalg.view<?xf32>
  %4 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  linalg.matvec(%2, %3, %4) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// TILE-2-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-2: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-2-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-2-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-2: affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.slice %[[A]][%[[ra]], %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-2-NEXT:   %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[rc:.*]] = linalg.range %i0:%[[c]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sCi:.*]] = linalg.slice %[[C]][%[[rc]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-2-NEXT:   linalg.matvec(%[[sAi]], %[[B]], %[[sCi]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-02-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-02: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-02-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-02-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-02: affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg2) step 2 {
//  TILE-02-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2{{.*}} : !linalg.range
//  TILE-02-NEXT:   %[[sAj:.*]] = linalg.slice %[[A]][%{{.*}}, %[[ra]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-02-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-02-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[b]]:%c2{{.*}} : !linalg.range
//  TILE-02-NEXT:   %[[sBj:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-02-NEXT:   linalg.matvec(%[[sAj]], %[[sBj]], %[[C]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

// TILE-002-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//   TILE-002-NOT: affine.for

// TILE-234-LABEL: func @matvec(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-234: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//  TILE-234-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-234-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       TILE-234:  affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-234-NEXT:    affine.for %i1 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg2) step 3 {
//  TILE-234-NEXT:      %[[ai:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[rai:.*]] = linalg.range %i0:%[[ai]]:%c2 : !linalg.range
//  TILE-234-NEXT:      %[[aj:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[raj:.*]] = linalg.range %i1:%[[aj]]:%c3 : !linalg.range
//  TILE-234-NEXT:      %[[sAij:.*]] = linalg.slice %[[A]][%[[rai]], %[[raj]]] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  TILE-234-NEXT:      %[[b:.*]] = affine.apply #[[UB1]](%i1)
//  TILE-234-NEXT:      %[[rb:.*]] = linalg.range %i1:%[[b]]:%c3 : !linalg.range
//  TILE-234-NEXT:      %[[sB:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-234-NEXT:      %[[c:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:      %[[rc:.*]] = linalg.range %i0:%[[c]]:%c2 : !linalg.range
//  TILE-234-NEXT:      %[[sC:.*]] = linalg.slice %[[C]][%[[rc]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-234-NEXT:      linalg.matvec(%[[sAij]], %[[sB]], %[[sC]]) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>

func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %1 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %2 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %3 = linalg.view %arg0[] : !linalg.view<f32>
  linalg.dot(%1, %2, %3) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// TILE-2-LABEL: func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-2: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-2-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-2-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<f32>
//       TILE-2: affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-2-NEXT:   %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sAi:.*]] = linalg.slice %[[A]][%[[ra]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-2-NEXT:   %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-2-NEXT:   %[[rb:.*]] = linalg.range %i0:%[[b]]:%c2 : !linalg.range
//  TILE-2-NEXT:   %[[sBi:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-2-NEXT:   linalg.dot(%[[sAi]], %[[sBi]], %[[C]]) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

// TILE-02-LABEL: func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//   TILE-02-NOT: affine.for

// TILE-002-LABEL: func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//   TILE-002-NOT: affine.for

// TILE-234-LABEL: func @dot(%arg0: !linalg.buffer<f32>, %arg1: index, %arg2: index, %arg3: index) {
//       TILE-234: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-234-NEXT: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//  TILE-234-NEXT: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<f32>
//       TILE-234:  affine.for %i0 = #[[ID]](%c0{{.*}}) to #[[ID]](%arg1) step 2 {
//  TILE-234-NEXT:    %[[a:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[ra:.*]] = linalg.range %i0:%[[a]]:%c2 : !linalg.range
//  TILE-234-NEXT:    %[[sA:.*]] = linalg.slice %[[A]][%[[ra]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-234-NEXT:    %[[b:.*]] = affine.apply #[[UB0]](%i0)
//  TILE-234-NEXT:    %[[rb:.*]] = linalg.range %i0:%[[b]]:%c2 : !linalg.range
//  TILE-234-NEXT:    %[[sB:.*]] = linalg.slice %[[B]][%[[rb]]] : !linalg.view<?xf32>, !linalg.range, !linalg.view<?xf32>
//  TILE-234-NEXT:    linalg.dot(%[[sA]], %[[sB]], %[[C]]) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
