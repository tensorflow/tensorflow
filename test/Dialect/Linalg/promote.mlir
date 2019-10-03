// RUN: mlir-opt %s -linalg-tile -linalg-tile-sizes=2,3,4 -linalg-tile-promote-full-tile-views=true | FileCheck %s -check-prefix=TILE-1D

// TILE-1D-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// TILE-1D-DAG: #[[strided2DnoOffset:.*]] = (d0, d1)[s0] -> (d0 * s0 + d1)

func @matmul(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %K = linalg.range %c0:%arg3:%c1 : !linalg.range
  %A = linalg.view %arg0[%I, %K] : !linalg.buffer<?xf32> -> memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = linalg.view %arg0[%K, %J] : !linalg.buffer<?xf32> -> memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = linalg.view %arg0[%I, %J] : !linalg.buffer<?xf32> -> memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// TILE-1D-LABEL: func @matmul(%{{.*}}: !linalg.buffer<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       TILE-1D:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       TILE-1D:         %[[vA:.*]] = linalg.subview {{.*}} : memref<?x?xf32, #[[strided2D]]>
//       TILE-1D:         %[[vB:.*]] = linalg.subview {{.*}} : memref<?x?xf32, #[[strided2D]]>
//       TILE-1D:         %[[vC:.*]] = linalg.subview {{.*}} : memref<?x?xf32, #[[strided2D]]>
///
//       TILE-1D:         %[[tmpA:.*]] = linalg.buffer_alloc : !linalg.buffer<8xf32>
//       TILE-1D:         %[[fullA:.*]] = linalg.view %[[tmpA]][{{.*}}] : !linalg.buffer<8xf32> -> memref<?x?xf32>
//       TILE-1D:         %[[partialA:.*]] = linalg.slice %[[fullA]][%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #[[strided2DnoOffset]]>
///
//       TILE-1D:         %[[tmpB:.*]] = linalg.buffer_alloc : !linalg.buffer<12xf32>
//       TILE-1D:         %[[fullB:.*]] = linalg.view %[[tmpB]][{{.*}}] : !linalg.buffer<12xf32> -> memref<?x?xf32>
//       TILE-1D:         %[[partialB:.*]] = linalg.slice %[[fullB]][%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #[[strided2DnoOffset]]>
///
//       TILE-1D:         %[[tmpC:.*]] = linalg.buffer_alloc : !linalg.buffer<6xf32>
//       TILE-1D:         %[[fullC:.*]] = linalg.view %[[tmpC]][{{.*}}] : !linalg.buffer<6xf32> -> memref<?x?xf32>
//       TILE-1D:         %[[partialC:.*]] = linalg.slice %[[fullC]][%{{.*}}, %{{.*}}] : memref<?x?xf32>, !linalg.range, !linalg.range, memref<?x?xf32, #[[strided2DnoOffset]]>

//       TILE-1D:         linalg.fill(%[[fullA]], {{.*}}) : memref<?x?xf32>, f32
//       TILE-1D:         linalg.fill(%[[fullB]], {{.*}}) : memref<?x?xf32>, f32
//       TILE-1D:         linalg.fill(%[[fullC]], {{.*}}) : memref<?x?xf32>, f32
//       TILE-1D:         linalg.copy(%[[vA]], %[[partialA]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2DnoOffset]]>
//       TILE-1D:         linalg.copy(%[[vB]], %[[partialB]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2DnoOffset]]>
//       TILE-1D:         linalg.copy(%[[vC]], %[[partialC]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2DnoOffset]]>
//
//       TILE-1D:         linalg.matmul(%[[fullA]], %[[fullB]], %[[fullC]]) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
//
//       TILE-1D:         linalg.copy(%[[partialC]], %[[vC]]) : memref<?x?xf32, #[[strided2DnoOffset]]>, memref<?x?xf32, #[[strided2D]]>
//
//       TILE-1D:         linalg.buffer_dealloc %[[tmpA]] : !linalg.buffer<8xf32>
//       TILE-1D:         linalg.buffer_dealloc %[[tmpB]] : !linalg.buffer<12xf32>
//       TILE-1D:         linalg.buffer_dealloc %[[tmpC]] : !linalg.buffer<6xf32>
