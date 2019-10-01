// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-DAG: #[[strided1D:.*]] = (d0)[s0] -> (d0 + s0)
#strided1D = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[strided2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
#strided2D = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
// CHECK-DAG: #[[strided3D:.*]] = (d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)
#strided3D = (d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)
// CHECK-DAG: #[[strided6D:.*]] = (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4 + d4 * s5 + d5 * s6)
#strided6D = (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4 + d4 * s5 + d5 * s6)

// CHECK-DAG: #[[map0:.*]] = (d0, d1, d2) -> (d0, d2, d1)
// CHECK-DAG: #[[map1:.*]] = (d0, d1, d2) -> (d2, d1, d0)

func @range(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range

func @buffer_size(%arg0: !linalg.buffer<?xf32>) -> index {
  %0 = linalg.buffer_size %arg0 : !linalg.buffer<?xf32>
  return %0 : index
}
// CHECK-LABEL: func @buffer_size
//       CHECK:   linalg.buffer_size {{.*}} : !linalg.buffer<?xf32>

func @buffer(%arg0: index, %arg1: index) {
  %0 = muli %arg0, %arg0 : index
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xvector<4xi8>>
  %2 = linalg.buffer_alloc %0 {alignment = 16} : !linalg.buffer<?xvector<4xi8>>
  %3 = linalg.buffer_alloc : !linalg.buffer<17xvector<4xi8>>
  %4 = linalg.buffer_alloc {alignment = 32} : !linalg.buffer<17xvector<4xi8>>
  linalg.buffer_dealloc %4 : !linalg.buffer<17xvector<4xi8>>
  linalg.buffer_dealloc %3 : !linalg.buffer<17xvector<4xi8>>
  linalg.buffer_dealloc %2 : !linalg.buffer<?xvector<4xi8>>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xvector<4xi8>>
  return
}
// CHECK-LABEL: func @buffer(%{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_alloc %{{.*}} {alignment = 16 : i64} : !linalg.buffer<?xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_alloc : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_alloc {alignment = 32 : i64} : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xvector<4xi8>>

func @views(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %0 = muli %arg0, %arg0 : index
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xf32>
  %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
  %3 = linalg.view %1[%2, %2] : !linalg.buffer<?xf32> -> memref<?x?xf32, #strided2D>
  %4 = linalg.slice %3[%2, %2] : memref<?x?xf32, #strided2D>, !linalg.range, !linalg.range, memref<?x?xf32, #strided2D>
  %5 = linalg.slice %3[%2, %arg2] : memref<?x?xf32, #strided2D>, !linalg.range, index, memref<?xf32, #strided1D>
  %6 = linalg.slice %3[%arg2, %2] : memref<?x?xf32, #strided2D>, index, !linalg.range, memref<?xf32, #strided1D>
  %7 = linalg.slice %3[%arg2, %arg3] : memref<?x?xf32, #strided2D>, index, index, memref<f32>
  %8 = linalg.view %1[%2, %2] : !linalg.buffer<?xf32> -> memref<?x?xvector<4x4xf32>, #strided2D>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xf32>
  return
}
// CHECK-LABEL: func @views(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xf32>
//  CHECK-NEXT:  linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range
//  CHECK-NEXT:  linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.buffer<?xf32> -> memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, !linalg.range, !linalg.range, memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, !linalg.range, index, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, index, !linalg.range, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32, #[[strided2D]]>, index, index, memref<f32>
//  CHECK-NEXT:  linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.buffer<?xf32> -> memref<?x?xvector<4x4xf32>, #[[strided2D]]>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xf32>

func @ops(%arg0: memref<?x?xf32, #strided2D>, %arg1: memref<?xf32, #strided1D>, %arg2: memref<?xf32, #strided1D>, %arg3: memref<f32>) {
  linalg.matmul(%arg0, %arg0, %arg0) : memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>
  linalg.matvec(%arg0, %arg1, %arg2) : memref<?x?xf32, #strided2D>, memref<?xf32, #strided1D>, memref<?xf32, #strided1D>
  linalg.dot(%arg1, %arg2, %arg3) : memref<?xf32, #strided1D>, memref<?xf32, #strided1D>, memref<f32>
  return
}
// CHECK-LABEL: func @ops(%
//       CHECK:  {{.*}}: memref<?x?xf32, #[[strided2D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<f32>) {
//  CHECK-NEXT:  linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:  linalg.matvec(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[strided2D]]>, memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>
//  CHECK-NEXT:  linalg.dot(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>, memref<f32>

func @dim(%arg0: memref<?x?xf32, #strided2D>) {
  %0 = dim %arg0, 1 : memref<?x?xf32, #strided2D>
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xf32>
  return
}
// CHECK-LABEL: func @dim(
//       CHECK:  %{{.*}}: memref<?x?xf32, #[[strided2D]]>) {
//  CHECK-NEXT:   dim %{{.*}}, 1 : memref<?x?xf32, #[[strided2D]]>
//  CHECK-NEXT:   linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xf32>
//  CHECK-NEXT:   linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xf32>

func @linalg_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    loop.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = cmpi "slt", %i0, %i1 : index
      %min = select %min_cmp, %i0, %i1 : index
      %max_cmp = cmpi "sge", %i0, %i1 : index
      %max = select %max_cmp, %i0, %i1 : index
      loop.for %i2 = %min to %max step %i1 {
      }
    }
  }
  return
}
// CHECK-LABEL: func @linalg_for(
//       CHECK:  %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:   loop.for %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     loop.for %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       loop.for %{{.*}} to %{{.*}} step %{{.*}} {

func @fill_view(%arg0: memref<?xf32, #strided1D>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?xf32, #strided1D>, f32
  return
}
// CHECK-LABEL: func @fill_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, f32

func @transpose(%arg0: memref<?x?x?xf32, #strided3D>) {
  %0 = linalg.transpose %arg0 (i, j, k) -> (k, j, i) : memref<?x?x?xf32, #strided3D>
  return
}
// CHECK-LABEL: func @transpose
//       CHECK:   linalg.transpose %{{.*}} ([[i:.*]], [[j:.*]], [[k:.*]]) -> ([[k]], [[j]], [[i]]) : memref<?x?x?xf32, #[[strided3D]]>

func @fill_view3(%arg0: memref<?x?x?xf32, #strided3D>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : memref<?x?x?xf32, #strided3D>, f32
  return
}
// CHECK-LABEL: func @fill_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : memref<?x?x?xf32, #[[strided3D]]>, f32

func @copy_view(%arg0: memref<?xf32, #strided1D>, %arg1: memref<?xf32, #strided1D>) {
  linalg.copy(%arg0, %arg1) : memref<?xf32, #strided1D>, memref<?xf32, #strided1D>
  return
}
// CHECK-LABEL: func @copy_view(
//       CHECK:  %{{.*}}: memref<?xf32, #[[strided1D]]>, %{{.*}}: memref<?xf32, #[[strided1D]]>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) : memref<?xf32, #[[strided1D]]>, memref<?xf32, #[[strided1D]]>

func @copy_view3(%arg0: memref<?x?x?xf32, #strided3D>, %arg1: memref<?x?x?xf32, #strided3D>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    memref<?x?x?xf32, #strided3D>, memref<?x?x?xf32, #strided3D>
  return
}
// CHECK-LABEL: func @copy_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) {inputPermutation = #[[map0]], outputPermutation = #[[map1]]} : memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>

func @conv_view3(%arg0: memref<?x?x?xf32, #strided3D>, %arg1: memref<?x?x?xf32, #strided3D>, %arg2: memref<?x?x?xf32, #strided3D>) {
  linalg.conv(%arg0, %arg1, %arg2) : memref<?x?x?xf32, #strided3D>, memref<?x?x?xf32, #strided3D>, memref<?x?x?xf32, #strided3D>
  return
}
// CHECK-LABEL: func @conv_view3(
//       CHECK:  %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>, %{{.*}}: memref<?x?x?xf32, #[[strided3D]]>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>, memref<?x?x?xf32, #[[strided3D]]>

func @conv_view6(%arg0: memref<?x?x?x?x?x?xf32, #strided6D>, %arg1: memref<?x?x?x?x?x?xf32, #strided6D>, %arg2: memref<?x?x?x?x?x?xf32, #strided6D>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : memref<?x?x?x?x?x?xf32, #strided6D>, memref<?x?x?x?x?x?xf32, #strided6D>, memref<?x?x?x?x?x?xf32, #strided6D>
  return
}
// CHECK-LABEL: func @conv_view6(
//       CHECK:  %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>, %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>, %{{.*}}: memref<?x?x?x?x?x?xf32, #[[strided6D]]>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : memref<?x?x?x?x?x?xf32, #[[strided6D]]>, memref<?x?x?x?x?x?xf32, #[[strided6D]]>, memref<?x?x?x?x?x?xf32, #[[strided6D]]>

func @subview(%arg0: memref<?x?xvector<3x4xi4>, #strided2D>) {
  %c0 = constant 0 : index
  %0 = linalg.subview %arg0[%c0, %c0, %c0, %c0, %c0, %c0] : memref<?x?xvector<3x4xi4>, #strided2D>
  return
}
// CHECK-LABEL: func @subview(
//       CHECK:  %{{.*}}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>) {
//       CHECK:   constant 0 : index
//       CHECK:   linalg.subview %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?xvector<3x4xi4>, #[[strided2D]]>

func @const_buffer_view(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = linalg.buffer_alloc : !linalg.buffer<17xf32>
  %c1 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  %c2 = linalg.view %c0[%c1] : !linalg.buffer<17xf32> -> memref<?xf32, #strided1D>
  return
}

#accesses = [
  (i, j, k) -> (j, i),
  (i, j, k) -> (i, k, i + j)
]
#trait = {
  indexing_maps = #accesses,
  n_views = [1, 1],
  n_loop_types = [3, 0, 0],
  fun = @foo,
  library_call = "external_function_name"
}
func @foo(%0: vector<3x4xi4>, %1: f32) -> f32 {
  %f0 = constant 0.0 : f32
  return %f0 : f32
}
func @generic(%arg0: memref<?x?xvector<3x4xi4>, #strided2D>, %arg1: memref<?x?x?xf32, #strided3D>) {
  linalg.generic #trait %arg0, %arg1 {foo = 1} : memref<?x?xvector<3x4xi4>, #strided2D>, memref<?x?x?xf32, #strided3D>
  return
}
// CHECK-LABEL: func @foo
// CHECK-LABEL: func @generic
//       CHECK:   linalg.generic {fun = @foo, indexing_maps = [#{{.*}}, #{{.*}}], library_call = "external_function_name", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {foo = 1 : i64}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>, memref<?x?x?xf32, #[[strided3D]]>

#trait2 = {
  indexing_maps = #accesses,
  n_views = [1, 1],
  n_loop_types = [3, 0, 0],
  library_call = "external_function_name"
}
func @generic_region(%arg0: memref<?x?xvector<3x4xi4>, #strided2D>, %arg1: memref<?x?x?xf32, #strided3D>) {
  linalg.generic #trait2 %arg0, %arg1 {
    ^bb(%a: vector<3x4xi4>, %b: f32) :
      linalg.yield %b : f32
  } {foo = 1}: memref<?x?xvector<3x4xi4>, #strided2D>, memref<?x?x?xf32, #strided3D>
  return
}
// CHECK-LABEL: func @generic_region
//       CHECK:   linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], library_call = "external_function_name", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {
//       CHECK:    ^{{.*}}(%{{.*}}: vector<3x4xi4>, %{{.*}}: f32):    // no predecessors
//       CHECK:      linalg.yield %{{.*}} : f32
//       CHECK:    } {foo = 1 : i64}: memref<?x?xvector<3x4xi4>, #[[strided2D]]>, memref<?x?x?xf32, #[[strided3D]]>
