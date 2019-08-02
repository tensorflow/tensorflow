// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: #[[map0:.*]] = (d0, d1, d2) -> (d0, d2, d1)
// CHECK: #[[map1:.*]] = (d0, d1, d2) -> (d2, d1, d0)

func @range(%arg0: index, %arg1: index, %arg2: index) {
  %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  return
}
// CHECK-LABEL: func @range(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  %{{.*}} = linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range

func @buffer(%arg0: index, %arg1: index) {
  %0 = muli %arg0, %arg0 : index
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xvector<4xi8>>
  %2 = linalg.buffer_alloc : !linalg.buffer<17xvector<4xi8>>
  linalg.buffer_dealloc %2 : !linalg.buffer<17xvector<4xi8>>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xvector<4xi8>>
  return
}
// CHECK-LABEL: func @buffer(%{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  %{{.*}} = muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  %{{.*}} = linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xvector<4xi8>>
//  CHECK-NEXT:  %{{.*}} = linalg.buffer_alloc : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<17xvector<4xi8>>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xvector<4xi8>>

func @view_fun(%arg0: !linalg.view<?x?xvector<3x4xi4>>) {
  return
}
// CHECK-LABEL: func @view_fun(%{{.*}}: !linalg.view<?x?xvector<3x4xi4>>) {

func @views(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %0 = muli %arg0, %arg0 : index
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xf32>
  %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
  %3 = linalg.view %1[%2, %2] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  %4 = linalg.slice %3[%2, %2] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
  %5 = linalg.slice %3[%2, %arg2] : !linalg.view<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
  %6 = linalg.slice %3[%arg2, %2] : !linalg.view<?x?xf32>, index, !linalg.range, !linalg.view<?xf32>
  %7 = linalg.slice %3[%arg2, %arg3] : !linalg.view<?x?xf32>, index, index, !linalg.view<f32>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xf32>
  return
}
// CHECK-LABEL: func @views(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  %{{.*}} = muli %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:  %{{.*}} = linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xf32>
//  CHECK-NEXT:  %{{.*}} = linalg.range %{{.*}}:%{{.*}}:%{{.*}} : !linalg.range
//  CHECK-NEXT:  %{{.*}} = linalg.view %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
//  CHECK-NEXT:  %{{.*}} = linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, !linalg.range, !linalg.view<?x?xf32>
//  CHECK-NEXT:  %{{.*}} = linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, !linalg.range, index, !linalg.view<?xf32>
//  CHECK-NEXT:  %{{.*}} = linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, index, !linalg.range, !linalg.view<?xf32>
//  CHECK-NEXT:  %{{.*}} = linalg.slice %{{.*}}[%{{.*}}, %{{.*}}] : !linalg.view<?x?xf32>, index, index, !linalg.view<f32>
//  CHECK-NEXT:  linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xf32>

func @ops(%arg0: !linalg.view<?x?xf32>, %arg1: !linalg.view<?xf32>, %arg2: !linalg.view<?xf32>, %arg3: !linalg.view<f32>) {
  linalg.matmul(%arg0, %arg0, %arg0) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
  linalg.matvec(%arg0, %arg1, %arg2) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
  linalg.dot(%arg1, %arg2, %arg3) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>
  return
}
// CHECK-LABEL: func @ops(%{{.*}}: !linalg.view<?x?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<f32>) {
//  CHECK-NEXT:  linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
//  CHECK-NEXT:  linalg.matvec(%{{.*}}, %{{.*}}, %{{.*}}) : !linalg.view<?x?xf32>, !linalg.view<?xf32>, !linalg.view<?xf32>
//  CHECK-NEXT:  linalg.dot(%{{.*}}, %{{.*}}, %{{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>, !linalg.view<f32>

func @dim(%arg0: !linalg.view<?x?xf32>) {
  %0 = linalg.dim %arg0, 1 : !linalg.view<?x?xf32>
  %1 = linalg.buffer_alloc %0 : !linalg.buffer<?xf32>
  linalg.buffer_dealloc %1 : !linalg.buffer<?xf32>
  return
}
// CHECK-LABEL: func @dim(%{{.*}}: !linalg.view<?x?xf32>) {
//  CHECK-NEXT:   %{{.*}} = linalg.dim %{{.*}}, 1 : !linalg.view<?x?xf32>
//  CHECK-NEXT:   %{{.*}} = linalg.buffer_alloc %{{.*}} : !linalg.buffer<?xf32>
//  CHECK-NEXT:   linalg.buffer_dealloc %{{.*}} : !linalg.buffer<?xf32>

func @range_intersect(%arg0: !linalg.range, %arg1: !linalg.range) -> !linalg.range {
  %0 = linalg.range_intersect %arg0, %arg1 : !linalg.range
  return %0 : !linalg.range
}
// CHECK-LABEL: func @range_intersect(%{{.*}}: !linalg.range, %{{.*}}: !linalg.range) -> !linalg.range {
//  CHECK-NEXT:   %{{.*}} = linalg.range_intersect %{{.*}}, %{{.*}} : !linalg.range
//  CHECK-NEXT:   return %{{.*}} : !linalg.range

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
// CHECK-LABEL: func @linalg_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:   loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:     loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//  CHECK-NEXT:       %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = cmpi "sge", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       %{{.*}} = select %{{.*}}, %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:       loop.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {

func @fill_view(%arg0: !linalg.view<?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : !linalg.view<?xf32>, f32

func @fill_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: f32) {
  linalg.fill(%arg0, %arg1) : !linalg.view<?x?x?xf32>, f32
  return
}
// CHECK-LABEL: func @fill_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: f32) {
//       CHECK:   linalg.fill(%{{.*}}, %{{.*}}) : !linalg.view<?x?x?xf32>, f32

func @copy_view(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>) {
  linalg.copy(%arg0, %arg1) : !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}
// CHECK-LABEL: func @copy_view(%{{.*}}: !linalg.view<?xf32>, %{{.*}}: !linalg.view<?xf32>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) : !linalg.view<?xf32>, !linalg.view<?xf32>

func @copy_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>) {
  linalg.copy(%arg0, %arg1) {inputPermutation = (i, j, k) -> (i, k, j),
                             outputPermutation = (i, j, k) -> (k, j, i)} :
    !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @copy_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>) {
//       CHECK:   linalg.copy(%{{.*}}, %{{.*}}) {inputPermutation = #[[map0]], outputPermutation = #[[map1]]} : !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>

func @conv_view3(%arg0: !linalg.view<?x?x?xf32>, %arg1: !linalg.view<?x?x?xf32>, %arg2: !linalg.view<?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) : !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view3(%{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?xf32>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) : !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>

func @conv_view6(%arg0: !linalg.view<?x?x?x?x?x?xf32>, %arg1: !linalg.view<?x?x?x?x?x?xf32>, %arg2: !linalg.view<?x?x?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : !linalg.view<?x?x?x?x?x?xf32>, !linalg.view<?x?x?x?x?x?xf32>, !linalg.view<?x?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func @conv_view6(%{{.*}}: !linalg.view<?x?x?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?x?x?xf32>, %{{.*}}: !linalg.view<?x?x?x?x?x?xf32>) {
//       CHECK:   linalg.conv(%{{.*}}, %{{.*}}, %{{.*}}) {dilations = [4, 4, 5, 5], strides = [2, 2, 3, 3]} : !linalg.view<?x?x?x?x?x?xf32>, !linalg.view<?x?x?x?x?x?xf32>, !linalg.view<?x?x?x?x?x?xf32>

func @subview(%arg0: !linalg.view<?x?xvector<3x4xi4>>) {
  %c0 = constant 0 : index
  %0 = linalg.subview %arg0[%c0, %c0, %c0, %c0, %c0, %c0] : !linalg.view<?x?xvector<3x4xi4>>
  return
}
// CHECK-LABEL: func @subview(%{{.*}}: !linalg.view<?x?xvector<3x4xi4>>) {
//       CHECK:   %{{.*}} = constant 0 : index
//       CHECK:   %{{.*}} = linalg.subview %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : !linalg.view<?x?xvector<3x4xi4>>

func @const_buffer_view(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = linalg.buffer_alloc : !linalg.buffer<17xf32>
  %c1 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
  %c2 = linalg.view %c0[%c1] : !linalg.buffer<17xf32> -> !linalg.view<?xf32>
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
func @generic(%arg0: !linalg.view<?x?xvector<3x4xi4>>, %arg1: !linalg.view<?x?x?xf32>) {
  linalg.generic #trait %arg0, %arg1 {foo = 1} : !linalg.view<?x?xvector<3x4xi4>>, !linalg.view<?x?x?xf32>
  return
}
// CHECK-LABEL: func @foo
// CHECK-LABEL: func @generic
//       CHECK:   linalg.generic {fun = @foo, indexing_maps = [#map2, #map3], library_call = "external_function_name", n_loop_types = [3, 0, 0], n_views = [1, 1]} %{{.*}}, %{{.*}} {foo = 1 : i64}: !linalg.view<?x?xvector<3x4xi4>>, !linalg.view<?x?x?xf32>
