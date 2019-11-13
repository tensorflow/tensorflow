// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// TODO(pifon): Re-enable LLVM lowering test after IndexedGenericOp is lowered.
//
// Test that we can lower all the way to LLVM without crashing, don't check results here.
// DISABLED: mlir-opt %s --convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// CHECK-DAG: #[[strided3D:.*]] = (d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)
// CHECK-DAG: #[[strided6D:.*]] = (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4 + d4 * s5 + d5)

// CHECK-DAG: #[[map0:.*]] = (d0, d1, d2) -> (d0, d2, d1)
// CHECK-DAG: #[[map1:.*]] = (d0, d1, d2) -> (d2, d1, d0)

func @range(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range

func @views(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %c0 = constant 0 : index
  %0 = muli %arg0, %arg0 : index
  %1 = alloc (%0) : memref<?xi8>
  %2 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  %3 = view %1[%arg0, %arg0][%c0] : memref<?xi8> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %4 = linalg.slice %3[%2, %2] : memref<?x?xf32, offset: ?, strides: [?, 1]>, !linalg.range, !linalg.range, memref<?x?xf32, offset: ?, strides: [?, 1]>
  %5 = linalg.slice %3[%2, %arg2] : memref<?x?xf32, offset: ?, strides: [?, 1]>, !linalg.range, index, memref<?xf32, offset: ?, strides: [1]>
  %6 = linalg.slice %3[%arg2, %2] : memref<?x?xf32, offset: ?, strides: [?, 1]>, index, !linalg.range, memref<?xf32, offset: ?, strides: [1]>
  %7 = linalg.slice %3[%arg2, %arg3] : memref<?x?xf32, offset: ?, strides: [?, 1]>, index, index, memref<f32>
  %8 = view %1[%arg0, %arg0][%c0] : memref<?xi8> to memref<?x?xvector<4x4xf32>, offset: ?, strides: [?, 1]>
  dealloc %1 : memref<?xi8>
  return
}
// CHECK-LABEL: func @views(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK:  muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  alloc(%{{.*}}) : memref<?xi8>
//  CHECK-NEXT:  range
//  CHECK-NEXT:  std.view %{{.*}}[%{{.*}}][%{{.*}}] : memref<?xi8> to memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, !linalg.range, !linalg.range, memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, !linalg.range, index, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, index, !linalg.range, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, index, index, memref<f32>
//  CHECK-NEXT:  view %{{.*}}[%{{.*}}][%{{.*}}] : memref<?xi8> to memref<?x?xvector<4x4xf32>, #[[strided2D]]>
//  CHECK-NEXT:  dealloc %{{.*}} : memref<?xi8>

func @ops(%arg0: memref<?x?xf32, offset: ?, strides: [?, 1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>, %arg2: memref<?xf32, offset: ?, strides: [1]>, %arg3: memref<f32>) {
  linalg.matmul(%arg0, %arg0, %arg0) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
  linalg.matvec(%arg0, %arg1, %arg2) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  linalg.dot(%arg1, %arg2, %arg3) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>, memref<f32>
  return
}
// CHECK-LABEL: func @ops(%
//       CHECK:  {{.*}}: memref<?x?xf32, #[[strided2D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//  CHECK-NEXT:  linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.matvec(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.dot(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>, memref<f32>

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, f32
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, f32

func @transpose(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  %0 = linalg.transpose %arg0 (i, j, k) -> (k, j, i) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   linalg.transpose %{{.*}} ([[i:.*]], [[j:.*]], [[k:.*]]) -> ([[k]], [[j]], [[i]]) : memref<?x?x?xf32, #[[strided3D]]>

func @fill_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, f32
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?x?x?xf32, #[[strided3D]]>, f32

func @copy_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, offset: ?, strides: [1]>, memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>

func @copy_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @copy_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) {inputPermutation = #[[map0]], outputPermutation = #[[map1]]} : memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, %arg2: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) : memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>

func @conv_view6(%arg0: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>, %arg1: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>, %arg2: memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>, memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>, memref<?x?x?x?x?x?xf32, offset: ?, strides: [?, ?, ?, ?, ?, 1]>
  return
}
// CHECK-LABEL: func @conv_view6(
//       CHECK:  %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>, %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>, %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : memref<?x?x?x?x?x?xf32, #[[strided6D]]>, memref<?x?x?x?x?x?xf32, #[[strided6D]]>, memref<?x?x?x?x?x?xf32, #[[strided6D]]>

#accesses = [
  (i, j, k) -> (j, i),
  (i, j, k) -> (i, k, i + j)
]
#trait = {
  indexing_maps = #accesses,
  n_views = [1, 1],
  n_loop_types = [3, 0, 0],
  fun = @foo,
  library_call = "some_external_function_name_1"
}
func @foo(%0: vector<3x4xi4>, %1: f32) -> f32 {
  %f0 = constant 0.0 : f32
  return %f0 : f32
}
func @generic(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait %arg0, %arg1 {foo = 1} : memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @foo
// CHECK-LABEL: func @generic
//       CHECK:   linalg.generic {fun = @foo, indexing_maps = [#{{.*}}, #{{.*}}], library_call = "some_external_function_name_1", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {foo = 1 : i64}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>, memref<?x?x?xf32, #[[strided3D]]>

#trait2 = {
  indexing_maps = #accesses,
  n_views = [1, 1],
  n_loop_types = [3, 0, 0],
  library_call = "some_external_function_name_2"
}
func @generic_region(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.generic #trait2 %arg0, %arg1 {
    ^bb(%a: vector<3x4xi4>, %b: f32) :
      linalg.yield %b : f32
  } {foo = 1}: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @generic_region
//       CHECK:   linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], library_call = "some_external_function_name_2", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {
//       CHECK:    ^{{.*}}(%{{.*}}: vector<3x4xi4>, %{{.*}}: f32):    // no predecessors
//       CHECK:      linalg.yield %{{.*}} : f32
//       CHECK:    } {foo = 1 : i64}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>, memref<?x?x?xf32, #[[strided3D]]>

func @indexed_generic(%arg0: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>,
                      %arg1: memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>) {
  linalg.indexed_generic #trait2 %arg0, %arg1 {
  ^bb(%i: index, %j: index, %k: index, %a: vector<3x4xi4>, %b: f32) :
      linalg.yield %b : f32
  } {foo = 1}: memref<?x?xvector<3x4xi4>, offset: ?, strides: [?, 1]>, memref<?x?x?xf32, offset: ?, strides: [?, ?, 1]>
  return
}
// CHECK-LABEL: func @indexed_generic
//       CHECK:   linalg.indexed_generic {indexing_maps = [#{{.*}}, #{{.*}}], library_call = "some_external_function_name_2", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {
//       CHECK:    ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: vector<3x4xi4>, %{{.*}}: f32):
//       CHECK:      linalg.yield %{{.*}} : f32
//       CHECK:    } {foo = 1 : i64}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>, memref<?x?x?xf32, #[[strided3D]]>
