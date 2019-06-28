// RUN: mlir-opt %s -linalg-lower-to-loops | FileCheck %s

// CHECK-DAG: #[[S2D1:.*]] = (d0, d1) -> (d0 * 2 + d1)
// CHECK-DAG: #[[S2D3:.*]] = (d0, d1) -> (d0 * 2 + d1 * 4)
// CHECK-DAG: #[[S3D2:.*]] = (d0, d1) -> (d0 * 3 + d1 * 5)

func @matmul(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
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
// CHECK-LABEL: func @matmul(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       CHECK: %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//       CHECK: linalg.for %i0 = %c0 to %[[M]] step %c1 {
//       CHECK:   linalg.for %i1 = %c0 to %[[N]] step %c1 {
//       CHECK:     linalg.for %i2 = %c0 to %[[K]] step %c1 {
//   CHECK-DAG:       %[[a:.*]] = linalg.load %[[A]][%i0, %i2] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[b:.*]] = linalg.load %[[B]][%i2, %i1] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = linalg.load %[[C]][%i0, %i1] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       linalg.store %[[res]], %[[C]][%i0, %i1] : !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
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
// CHECK-LABEL: func @matvec(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       CHECK: linalg.for %i0 = %c0 to %[[M]] step %c1 {
//       CHECK:   linalg.for %i1 = %c0 to %[[K]] step %c1 {
//   CHECK-DAG:     %[[a:.*]] = linalg.load %[[A]][%i0, %i1] : !linalg.view<?x?xf32>
//   CHECK-DAG:     %[[b:.*]] = linalg.load %[[B]][%i1] : !linalg.view<?xf32>
//   CHECK-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:     %[[c:.*]] = linalg.load %[[C]][%i0] : !linalg.view<?xf32>
//   CHECK-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     linalg.store %[[res]], %[[C]][%i0] : !linalg.view<?xf32>

func @dot(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %1 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %2 = linalg.view %arg0[%I] : !linalg.view<?xf32>
  %3 = linalg.view %arg0[] : !linalg.view<f32>
  linalg.dot(%1, %2, %3) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[] : !linalg.view<f32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?xf32>
//       CHECK: linalg.for %i0 = %c0 to %[[K]] step %c1 {
//   CHECK-DAG:   %[[a:.*]] = linalg.load %[[A]][%i0] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[b:.*]] = linalg.load %[[B]][%i0] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = linalg.load %[[C]][] : !linalg.view<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   linalg.store %[[res]], %[[C]][] : !linalg.view<f32>

func @dot_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
//       CHECK: %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       CHECK: linalg.for %i0 = %c0 to %[[K]] step %c1 {
//   CHECK-DAG:   %[[a:.*]] = linalg.load %arg0[%i0] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[b:.*]] = linalg.load %arg1[%i0] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = linalg.load %arg2[] : !linalg.view<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   linalg.store %[[res]], %arg2[] : !linalg.view<f32>

func @fill_view(%arg0: !linalg.view<?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view(%arg0: !linalg.view<?xf32>, %arg1: f32) {
//       CHECK:   linalg.for %i0 = %c0 to %0 step %c1 {
//       CHECK:     linalg.store %arg1, %arg0[%i0] : !linalg.view<?xf32>

func @fill_view0(%arg0: !linalg.view<f32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<f32>, f32
  return
}
// CHECK-LABEL: func @fill_view0(%arg0: !linalg.view<f32>, %arg1: f32) {
//       CHECK:   linalg.store %arg1, %arg0[] : !linalg.view<f32>

func @fill_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?x?x?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: f32) {
//       CHECK:   linalg.for %i0 = %c0 to %{{.*}} step %c1 {
//       CHECK:     linalg.for %i1 = %c0 to %{{.*}} step %c1 {
//       CHECK:       linalg.for %i2 = %c0 to %{{.*}} step %c1 {
//       CHECK:         linalg.store %arg1, %arg0[%i0, %i1, %i2] : !linalg.view<?x?x?xf32>

func @copy_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>) {
  linalg.copy(%arg0, %arg1) : !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @copy_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>) {
//       CHECK:   linalg.for %i0 = %c0 to %0 step %c1 {
//       CHECK:     %[[L:.*]] = linalg.load %arg0[%i0] : !linalg.view<?xf32>
//       CHECK:     linalg.store %[[L]], %arg1[%i0] : !linalg.view<?xf32>

func @copy_view0(%arg0: !linalg.view<f32>, %arg1: !linalg.view<f32>) {
  linalg.copy(%arg0, %arg1) : !linalg.view<f32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @copy_view0(%arg0: !linalg.view<f32>, %arg1: !linalg.view<f32>) {
//       CHECK:   %0 = linalg.load %arg0[] : !linalg.view<f32>
//       CHECK:   linalg.store %0, %arg1[] : !linalg.view<f32>

func @copy_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @copy_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>) {
//       CHECK:   linalg.for %i0 = %c0 to %{{.*}} step %c1 {
//       CHECK:     linalg.for %i1 = %c0 to %{{.*}} step %c1 {
//       CHECK:       linalg.for %i2 = %c0 to %{{.*}} step %c1 {
//       CHECK:         %[[L:.*]] = linalg.load %arg0[%i0, %i2, %i1] : !linalg.view<?x?x?xf32>
//       CHECK:         linalg.store %[[L]], %arg1[%i2, %i1, %i0] : !linalg.view<?x?x?xf32>

func @conv_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>, %arg2: !linalg.view<?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>, %arg2: !linalg.view<?x?x?xf32>) {
//       CHECK:   %[[Z0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[Q:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[K:.*]] = linalg.dim %arg0, 2 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[B:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[X0:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?x?xf32>
//       CHECK:   linalg.for %i0 = %c0 to %[[B]] step %c1 {
//       CHECK:     linalg.for %i1 = %c0 to %[[X0]] step %c1 {
//       CHECK:       linalg.for %i2 = %c0 to %[[K]] step %c1 {
//       CHECK:         linalg.for %i3 = %c0 to %[[Q]] step %c1 {
//       CHECK:           linalg.for %i4 = %c0 to %[[Z0]] step %c1 {
//       CHECK:             %[[SUM:.*]] = affine.apply #[[S2D1]](%i1, %i4)
//       CHECK:             %{{.*}} = linalg.load %arg1[%i0, %[[SUM]], %i3] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = linalg.load %arg0[%i4, %i3, %i2] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:             %{{.*}} = linalg.load %arg2[%i0, %i1, %i2] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:             linalg.store %{{.*}}, %arg2[%i0, %i1, %i2] : !linalg.view<?x?x?xf32>

func @conv_view4(%arg0: !linalg.view<?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view4(%arg0: !linalg.view<?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?xf32>) {
//       CHECK:   %[[Z0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[Z1:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[Q:.*]] = linalg.dim %arg0, 2 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[K:.*]] = linalg.dim %arg0, 3 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[B:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[X0:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[X1:.*]] = linalg.dim %arg2, 2 : !linalg.view<?x?x?x?xf32>
//       CHECK:   linalg.for %i0 = %c0 to %[[B]] step %c1 {
//       CHECK:     linalg.for %i1 = %c0 to %[[X0]] step %c1 {
//       CHECK:       linalg.for %i2 = %c0 to %[[X1]] step %c1 {
//       CHECK:         linalg.for %i3 = %c0 to %[[K]] step %c1 {
//       CHECK:           linalg.for %i4 = %c0 to %[[Q]] step %c1 {
//       CHECK:             linalg.for %i5 = %c0 to %[[Z0]] step %c1 {
//       CHECK:               linalg.for %i6 = %c0 to %[[Z1]] step %c1 {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #map1(%i1, %i5)
//       CHECK:                 %[[SUM1:.*]] = affine.apply #map2(%i2, %i6)
//       CHECK:                 %{{.*}} = linalg.load %arg1[%i0, %[[SUM0]], %[[SUM1]], %i4] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = linalg.load %arg0[%i5, %i6, %i4, %i3] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = linalg.load %arg2[%i0, %i1, %i2, %i3] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:                 linalg.store %{{.*}}, %arg2[%i0, %i1, %i2, %i3] : !linalg.view<?x?x?x?xf32>
