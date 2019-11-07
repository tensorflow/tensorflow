// RUN: mlir-opt %s -linalg-lower-to-loops | FileCheck %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// CHECK-DAG: #[[strided3D:.*]] = (d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)
// CHECK-DAG: #[[strided4D:.*]] = (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)

// CHECK-DAG: #[[Stride2Dilation1:.*]] = (d0, d1) -> (d0 * 2 + d1)
// CHECK-DAG: #[[Stride2Dilation4:.*]] = (d0, d1) -> (d0 * 2 + d1 * 4)
// CHECK-DAG: #[[Stride3Dilation5:.*]] = (d0, d1) -> (d0 * 3 + d1 * 5)


func @matmul(%arg0: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %A = view %arg0[%M, %K][%c0] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %B = view %arg0[%K, %N][%c0] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %C = view %arg0[%M, %N][%c0] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[M:.*]] = dim %[[A]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[K:.*]] = dim %[[A]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[N:.*]] = dim %[[B]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:       %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[b:.*]] = load %[[B]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:       %[[c:.*]] = load %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:       %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:       store %[[res]], %[[C]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>

func @matvec(%arg0: memref<?xi8>, %M: index, %N: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %2 = view %arg0[%M, %N][%c0] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %3 = view %arg0[%M][%c0] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %4 = view %arg0[%N][%c0] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  linalg.matvec(%2, %3, %4) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @matvec(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[M:.*]] = dim %[[A]], 0 : memref<?x?xf32, #[[strided2D]]>
//       CHECK: %[[K:.*]] = dim %[[A]], 1 : memref<?x?xf32, #[[strided2D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[M]] step %{{.*}} {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:     %[[a:.*]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>
//   CHECK-DAG:     %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:     %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:     %[[c:.*]] = load %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:     %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:     store %[[res]], %[[C]][%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @dot(%arg0: memref<?xi8>, %M: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %1 = view %arg0[%M][%c0] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %2 = view %arg0[%M][%c0] : memref<?xi8> to memref<?xf32, offset: ?, strides: [1]>
  %3 = view %arg0[][] : memref<?xi8> to memref<f32>
  linalg.dot(%1, %2, %3) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @dot(%{{.*}}: memref<?xi8>, %{{.*}}: index) {
//       CHECK: %[[A:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[B:.*]] = std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?xf32, #[[strided1D]]>
//       CHECK: %[[C:.*]] = std.view %{{.*}}[][] : memref<?xi8> to memref<f32>
//       CHECK: %[[K:.*]] = dim %[[A]], 0 : memref<?xf32, #[[strided1D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = load %[[A]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[b:.*]] = load %[[B]][%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = load %[[C]][] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %[[C]][] : memref<f32>

func @dot_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<f32>) {
  linalg.dot(%arg0, %arg1, %arg2) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @dot_view(
//       CHECK:   %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//       CHECK: %[[K:.*]] = dim %arg0, 0 : memref<?xf32, #[[strided1D]]>
//       CHECK: loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//   CHECK-DAG:   %[[a:.*]] = load %arg0[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[b:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//   CHECK-DAG:   %[[inc:.*]] = mulf %[[a]], %[[b]] : f32
//   CHECK-DAG:   %[[c:.*]] = load %{{.*}}[] : memref<f32>
//   CHECK-DAG:   %[[res:.*]] = addf %[[c]], %[[inc]] : f32
//       CHECK:   store %[[res]], %{{.*}}[] : memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @fill_view0(%arg0: memref<f32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<f32>, f32
  return
}
// CHECK-LABEL: func @fill_view0(%{{.*}}: memref<f32>, %{{.*}}: f32) {
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, f32
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK: %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     %[[L:.*]] = load %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>
//       CHECK:     store %[[L]], %{{.*}}[%{{.*}}] : memref<?xf32, #[[strided1D]]>

func @copy_view0(%arg0: memref<f32>, %arg1: memref<f32>) {
  linalg.copy(%arg0, %arg1) : memref<f32>, memref<f32>
  return
}
// CHECK-LABEL: func @copy_view0(%{{.*}}: memref<f32>, %{{.*}}: memref<f32>) {
//       CHECK:   %{{.*}} = load %{{.*}}[] : memref<f32>
//       CHECK:   store %{{.*}}, %{{.*}}[] : memref<f32>

func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_view3
//       CHECK: (%{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[L:.*]] = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:         store %[[L]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {strides = [2]}: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK: %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[Q:.*]] = dim %arg0, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[K:.*]] = dim %arg0, 2 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:         loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:           loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:             %[[SUM:.*]] = affine.apply #[[Stride2Dilation1]](%{{.*}}, %{{.*}})
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM]], %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:             %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:             store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32, #[[strided3D]]>

func @conv_view4(%arg0: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg1: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, %arg2: memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 5], strides = [2, 3]} : memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>, memref<?x?x?x?xf32, offset: ?, strides: [?, ?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view4(
//       CHECK: %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>, %{{.*}}: memref<?x?x?x?xf32, #[[strided4D]]>) {
//       CHECK:   %[[Z0:.*]] = dim %arg0, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[Z1:.*]] = dim %arg0, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[Q:.*]] = dim %arg0, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[K:.*]] = dim %arg0, 3 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[B:.*]] = dim %arg1, 0 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[X0:.*]] = dim %arg2, 1 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   %[[X1:.*]] = dim %arg2, 2 : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:   loop.for %{{.*}} = %{{.*}} to %[[B]] step %{{.*}} {
//       CHECK:     loop.for %{{.*}} = %{{.*}} to %[[X0]] step %{{.*}} {
//       CHECK:       loop.for %{{.*}} = %{{.*}} to %[[X1]] step %{{.*}} {
//       CHECK:         loop.for %{{.*}} = %{{.*}} to %[[K]] step %{{.*}} {
//       CHECK:           loop.for %{{.*}} = %{{.*}} to %[[Q]] step %{{.*}} {
//       CHECK:             loop.for %{{.*}} = %{{.*}} to %[[Z0]] step %{{.*}} {
//       CHECK:               loop.for %{{.*}} = %{{.*}} to %[[Z1]] step %{{.*}} {
//       CHECK:                 %[[SUM0:.*]] = affine.apply #[[Stride2Dilation4]](%{{.*}}, %{{.*}})
//       CHECK:                 %[[SUM1:.*]] = affine.apply #[[Stride3Dilation5]](%{{.*}}, %{{.*}})
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %[[SUM0]], %[[SUM1]], %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
//       CHECK:                 %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>
//       CHECK:                 %{{.*}} = addf %{{.*}}, %{{.*}} : f32
//       CHECK:                 store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?x?xf32, #[[strided4D]]>

func @foo(%0: f32, %1: f32, %2: f32) -> (f32, f32) {
  %f0 = constant 0.0 : f32
  return %f0, %f0 : f32, f32
}
#accesses = [
  (i, j, k) -> (i, j),
  (i, j, k) -> (i, j, k),
  (i, j, k) -> (i, k, j)
]
#trait = {
  n_views = [1, 2],
  n_loop_types = [3, 0, 0],
  indexing_maps = #accesses,
  fun = @foo,
  library_call = "some_external_function_name_1",
  doc = "B(i,j,k), C(i,k,j) = foo(A(i, j), B(i,j,k), C(i,k,j))"
}
func @generic_function(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait %arg0, %arg1, %arg2:
    memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: @foo
// CHECK-LABEL: @generic_function
//       CHECK: loop.for %[[i:.*]] = {{.*}}
//       CHECK:   loop.for %[[j:.*]] = {{.*}}
//       CHECK:     loop.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[res:.*]]:2 = call @foo(%[[a]], %[[b]], %[[c]]) : (f32, f32, f32) -> (f32, f32)
//       CHECK:       store %[[res]]#0, %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       store %[[res]]#1, %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>

#trait2 = {
  n_views = [1, 2],
  n_loop_types = [3, 0, 0],
  indexing_maps = #accesses,
  library_call = "some_external_function_name_2",
  doc = "B(i,j,k), C(i,k,j) = foo(A(i, j), B(i,j,k), C(i,k,j))"
}
func @generic_region(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait2 %arg0, %arg1, %arg2 {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %d = mulf %a, %b : f32
      %e = addf %c, %d : f32
      linalg.yield %d, %e : f32, f32
  }: memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: @generic_region
//       CHECK: loop.for %[[i:.*]] = {{.*}}
//       CHECK:   loop.for %[[j:.*]] = {{.*}}
//       CHECK:     loop.for %[[k:.*]] = {{.*}}
//       CHECK:       %[[a:.*]] = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32, #[[strided2D]]>
//       CHECK:       %[[b:.*]] = load %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[c:.*]] = load %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       %[[d:.*]] = mulf %[[a]], %[[b]] : f32
//       CHECK:       %[[e:.*]] = addf %[[c]], %[[d]] : f32
//       CHECK:       store %[[d]], %{{.*}}[%[[i]], %[[j]], %[[k]]] : memref<?x?x?xf32, #[[strided3D]]>
//       CHECK:       store %[[e]], %{{.*}}[%[[i]], %[[k]], %[[j]]] : memref<?x?x?xf32, #[[strided3D]]>
