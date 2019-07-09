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
  %A = linalg.view %arg0[%I, %K] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %B = linalg.view %arg0[%K, %J] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %C = linalg.view %arg0[%I, %J] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  linalg.matmul(%A, %B, %C) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  return
}
// CHECK-LABEL: func @matmul(%{{.*}}: !linalg.buffer<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
//       CHECK: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       CHECK: %[[N:.*]] = linalg.dim %[[B]], 1 : !linalg.view<?x?xf32>
//       CHECK: linalg.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECK:     linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:       %[[a:.*]] = linalg.load %[[A]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[b:.*]] = linalg.load %[[B]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = linalg.load %[[C]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       linalg.store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>

func @matvec(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %J = linalg.range %c0:%arg2:%c1 : !linalg.range
  %2 = linalg.view %arg0[%I, %J] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %3 = linalg.view %arg0[%J] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %4 = linalg.view %arg0[%I] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  linalg.matvec(%2, %3, %4) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @matvec(%{{.*}}: !linalg.buffer<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
//       CHECK: %[[M:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?x?xf32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 1 : !linalg.view<?x?xf32>
//       CHECK: linalg.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:     %[[a:.*]] = linalg.load %[[A]][%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>
//   CHECK-DAG:     %[[b:.*]] = linalg.load %[[B]][%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:     %[[c:.*]] = linalg.load %[[C]][%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     linalg.store %[[res]], %[[C]][%{{.*}}] : !linalg.view<?xf32>

func @dot(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %I = linalg.range %c0:%arg1:%c1 : !linalg.range
  %1 = linalg.view %arg0[%I] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %2 = linalg.view %arg0[%I] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
  %3 = linalg.view %arg0[] : !linalg.buffer<?xf32> -> !linalg.view<f32>
  linalg.dot(%1, %2, %3) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot(%{{.*}}: !linalg.buffer<?xf32>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
//       CHECK: %[[B:.*]] = linalg.view %arg0[{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?xf32>
//       CHECK: %[[C:.*]] = linalg.view %arg0[] : !linalg.buffer<?xf32> -> !linalg.view<f32>
//       CHECK: %[[K:.*]] = linalg.dim %[[A]], 0 : !linalg.view<?xf32>
//       CHECK: linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = linalg.load %[[A]][%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[b:.*]] = linalg.load %[[B]][%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = linalg.load %[[C]][] : !linalg.view<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   linalg.store %[[res]], %[[C]][] : !linalg.view<f32>

func @dot_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @dot_view(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//       CHECK: %[[K:.*]] = linalg.dim %arg0, 0 : !linalg.view<?xf32>
//       CHECK: linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = linalg.load %arg0[%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[b:.*]] = linalg.load %{{.*}}[%{{.*}}] : !linalg.view<?xf32>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = linalg.load %{{.*}}[] : !linalg.view<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   linalg.store %[[res]], %{{.*}}[] : !linalg.view<f32>

func @fill_view(%arg0: !linalg.view<?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: f32) {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     linalg.store %{{.*}}, %{{.*}}[%{{.*}}] : !linalg.view<?xf32>

func @fill_view0(%arg0: !linalg.view<f32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<f32>, f32
  return
}
// CHECK-LABEL: func @fill_view0(%{{.*}}: !linalg.view<f32>, %{{.*}}: f32) {
//       CHECK:   linalg.store %{{.*}}, %{{.*}}[] : !linalg.view<f32>

func @fill_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?x?x?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: f32) {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         linalg.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>

func @copy_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>) {
  linalg.copy(%arg0, %arg1) : !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @copy_view(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     %[[L:.*]] = linalg.load %{{.*}}[%{{.*}}] : !linalg.view<?xf32>
//       CHECK:     linalg.store %[[L]], %{{.*}}[%{{.*}}] : !linalg.view<?xf32>

func @copy_view0(%arg0: !linalg.view<f32>, %arg1: !linalg.view<f32>) {
  linalg.copy(%arg0, %arg1) : !linalg.view<f32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @copy_view0(%{{.*}}: !linalg.view<f32>, %{{.*}}: !linalg.view<f32>) {
//       CHECK:   %{{.*}} = linalg.load %{{.*}}[] : !linalg.view<f32>
//       CHECK:   linalg.store %{{.*}}, %{{.*}}[] : !linalg.view<f32>

func @copy_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @copy_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>) {
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       linalg.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[L:.*]] = linalg.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>
//       CHECK:         linalg.store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>

func @conv_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>, %arg2: !linalg.view<?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>) {
//       CHECK:   %[[Z0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[Q:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[K:.*]] = linalg.dim %arg0, 2 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[B:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?x?xf32>
//       CHECK:   %[[X0:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?x?xf32>
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     linalg.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:         linalg.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:           linalg.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:             %[[SUM:.*]] = affine.apply #[[S2D1]](%{{.*}}, %{{.*}})
//       CHECK:             %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:             %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>
//       CHECK:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:             linalg.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?xf32>

func @conv_view4(%arg0: !linalg.view<?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>, !linalg.view<?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view4(%{{.*}}: !linalg.view<?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?xf32>) {
//       CHECK:   %[[Z0:.*]] = linalg.dim %arg0, 0 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[Z1:.*]] = linalg.dim %arg0, 1 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[Q:.*]] = linalg.dim %arg0, 2 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[K:.*]] = linalg.dim %arg0, 3 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[B:.*]] = linalg.dim %arg1, 0 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[X0:.*]] = linalg.dim %arg2, 1 : !linalg.view<?x?x?x?xf32>
//       CHECK:   %[[X1:.*]] = linalg.dim %arg2, 2 : !linalg.view<?x?x?x?xf32>
//       CHECK:   linalg.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     linalg.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       linalg.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECK:         linalg.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:           linalg.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:             linalg.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:               linalg.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #map1(%{{.*}}, %{{.*}})
//       CHECK:                 %[[SUM1:.*]] = affine.apply #map2(%{{.*}}, %{{.*}})
//       CHECK:                 %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = linalg.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?x?xf32>
//       CHECK:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:                 linalg.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?x?x?xf32>
