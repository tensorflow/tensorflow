// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 -linalg-tile-promote-full-tile-views=true | FileCheck %s -check-prefix=TILE-1D

func @matmul(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %K = linalg.range %c0:%arg3:%c1 : !linalg.range
  %A = linalg.view %arg0[%I, %K] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %B = linalg.view %arg0[%K, %J] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %C = linalg.view %arg0[%I, %J] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// TILE-1D-LABEL: func @matmul(%{{.*}}: !linalg.buffer<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       TILE-1D:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:         %[[vA:.*]] = linalg.subview {{.*}} : !linalg.view<?x?xf32>
//       TILE-1D:         %[[vB:.*]] = linalg.subview {{.*}} : !linalg.view<?x?xf32>
//       TILE-1D:         %[[vC:.*]] = linalg.subview {{.*}} : !linalg.view<?x?xf32>
///
//       TILE-1D:         %[[tmpA:.*]] = linalg.buffer_alloc : !linalg.buffer<8xf32>
//       TILE-1D:         %[[fullA:.*]] = linalg.view %[[tmpA]][{{.*}}] : !linalg.buffer<8xf32> -> !linalg.view<?x?xf32>
//       TILE-1D:         %[[partialA:.*]] = linalg.slice %[[fullA]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
///
//       TILE-1D:         %[[tmpB:.*]] = linalg.buffer_alloc : !linalg.buffer<12xf32>
//       TILE-1D:         %[[fullB:.*]] = linalg.view %[[tmpB]][{{.*}}] : !linalg.buffer<12xf32> -> !linalg.view<?x?xf32>
//       TILE-1D:         %[[partialB:.*]] = linalg.slice %[[fullB]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
///
//       TILE-1D:         %[[tmpC:.*]] = linalg.buffer_alloc : !linalg.buffer<6xf32>
//       TILE-1D:         %[[fullC:.*]] = linalg.view %[[tmpC]][{{.*}}] : !linalg.buffer<6xf32> -> !linalg.view<?x?xf32>
//       TILE-1D:         %[[partialC:.*]] = linalg.slice %[[fullC]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>

//       TILE-1D:         linalg.fill(%[[fullA]], {{.*}}) : !linalg.view<?x?xf32>, f32
//       TILE-1D:         linalg.fill(%[[fullB]], {{.*}}) : !linalg.view<?x?xf32>, f32
//       TILE-1D:         linalg.fill(%[[fullC]], {{.*}}) : !linalg.view<?x?xf32>, f32
//       TILE-1D:         linalg.copy(%[[vA]], %[[partialA]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//       TILE-1D:         linalg.copy(%[[vB]], %[[partialB]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//       TILE-1D:         linalg.copy(%[[vC]], %[[partialC]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//
//       TILE-1D:         linalg.matmul(%[[fullA]], %[[fullB]], %[[fullC]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//
//       TILE-1D:         linalg.copy(%[[partialC]], %[[vC]]) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//
//       TILE-1D:         linalg.buffer_dealloc %[[tmpA]] : !linalg.buffer<8xf32>
//       TILE-1D:         linalg.buffer_dealloc %[[tmpB]] : !linalg.buffer<12xf32>
//       TILE-1D:         linalg.buffer_dealloc %[[tmpC]] : !linalg.buffer<6xf32>
