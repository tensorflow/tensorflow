// RUN: mlir-opt %s -test-linalg-transform-patterns | FileCheck %s

// CHECK-DAG: #[[STRIDED_1D:.*]] = (d0)[s0] -> (d0 + s0)
// CHECK-DAG: #[[STRIDED_2D:.*]] = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)

func @dot(%x: memref<?xf32, offset: ?, strides: [1]>,
          %y: memref<?xf32, offset: ?, strides: [1]>,
          %v: memref<f32>) {
  linalg.dot(%x, %y, %v) : memref<?xf32, offset: ?, strides: [1]>,
                           memref<?xf32, offset: ?, strides: [1]>,
                           memref<f32>
  return
}
// CHECK-LABEL: func @dot
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c8:.*]] = constant 8 : index
// CHECK-DAG  :   %[[c8000:.*]] = constant 8000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c8]] {
// CHECK      :       linalg.dot({{.*}}, {{.*}}, {{.*}}) : memref<?xf32, #[[STRIDED_1D]]>, memref<?xf32, #[[STRIDED_1D]]>, memref<f32>

func @matvec(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %x: memref<?xf32, offset: ?, strides: [1]>,
             %y: memref<?xf32, offset: ?, strides: [1]>) {
  linalg.matvec(%A, %x, %y) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?xf32, offset: ?, strides: [1]>,
                              memref<?xf32, offset: ?, strides: [1]>
  return
}
// CHECK-LABEL: func @matvec
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c5:.*]] = constant 5 : index
// CHECK-DAG  :   %[[c6:.*]] = constant 6 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c5]]
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c6]]
// CHECK      :       linalg.matvec({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?xf32, #[[STRIDED_1D]]>, memref<?xf32, #[[STRIDED_1D]]>

func @matmul(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
             %C: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>
  return
}
// CHECK-LABEL: func @matmul
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c2:.*]] = constant 2 : index
// CHECK-DAG  :   %[[c3:.*]] = constant 3 : index
// CHECK-DAG  :   %[[c4:.*]] = constant 4 : index
// CHECK-DAG  :   %[[c20:.*]] = constant 20 : index
// CHECK-DAG  :   %[[c30:.*]] = constant 30 : index
// CHECK-DAG  :   %[[c40:.*]] = constant 40 : index
// CHECK-DAG  :   %[[c200:.*]] = constant 200 : index
// CHECK-DAG  :   %[[c300:.*]] = constant 300 : index
// CHECK-DAG  :   %[[c400:.*]] = constant 400 : index
// CHECK-DAG  :   %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG  :   %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG  :   %[[c4000:.*]] = constant 4000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK      :           loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK      :             loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK      :               loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK      :                 loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK      :                   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK      :                     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK      :                       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK      :                         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK      :                           linalg.matmul({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

#some_generic_trait = {
  indexing_maps = [
    (i, j) -> (i, j),
    (i, j) -> (i, j)
  ],
  n_views = [1, 1],
  n_loop_types = [2, 0, 0]
}
func @fusion_test(%A: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %B: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %C: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %D: memref<?x?xf32, offset: ?, strides: [?, 1]>,
                  %E: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  // This should not be fused as it would violate dependencies. It will get
  // tiled for all levels of the memory hierarchy.
  linalg.matmul(%A, %A, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This should be fused.
  linalg.matmul(%A, %B, %C) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  // This should not be fused or transformed at all since there are no patterns
  // on it. However it will be reordered because there are no dependencies.
  linalg.generic #some_generic_trait %A, %D {
    ^bb(%a: f32, %b: f32) :
      linalg.yield %a : f32
  } : memref<?x?xf32, offset: ?, strides: [?, 1]>,
      memref<?x?xf32, offset: ?, strides: [?, 1]>

  linalg.matmul(%C, %D, %E) : memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>,
                              memref<?x?xf32, offset: ?, strides: [?, 1]>

  return
}
// CHECK-LABEL: func @fusion_test
// CHECK-DAG  :   %[[c0:.*]] = constant 0 : index
// CHECK-DAG  :   %[[c2:.*]] = constant 2 : index
// CHECK-DAG  :   %[[c3:.*]] = constant 3 : index
// CHECK-DAG  :   %[[c4:.*]] = constant 4 : index
// CHECK-DAG  :   %[[c20:.*]] = constant 20 : index
// CHECK-DAG  :   %[[c30:.*]] = constant 30 : index
// CHECK-DAG  :   %[[c40:.*]] = constant 40 : index
// CHECK-DAG  :   %[[c100:.*]] = constant 100 : index
// CHECK-DAG  :   %[[c150:.*]] = constant 150 : index
// CHECK-DAG  :   %[[c200:.*]] = constant 200 : index
// CHECK-DAG  :   %[[c300:.*]] = constant 300 : index
// CHECK-DAG  :   %[[c400:.*]] = constant 400 : index
// CHECK-DAG  :   %[[c2000:.*]] = constant 2000 : index
// CHECK-DAG  :   %[[c3000:.*]] = constant 3000 : index
// CHECK-DAG  :   %[[c4000:.*]] = constant 4000 : index
// CHECK      :   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2000]] {
// CHECK      :     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3000]] {
// CHECK      :       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4000]] {
// CHECK      :         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c200]] {
// CHECK      :           loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c300]] {
// CHECK      :             loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c400]] {
// CHECK      :               loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c20]] {
// CHECK      :                 loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c30]] {
// CHECK      :                   loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c40]] {
// CHECK      :                     loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c2]] {
// CHECK      :                       loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c3]] {
// CHECK      :                         loop.for {{.*}} = %[[c0]] to {{.*}} step %[[c4]] {
// CHECK      :                           linalg.matmul({{.*}}, {{.*}}, {{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>
//
// CHECK      :   linalg.generic
//
// CHECK      :   loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c100]] {
// CHECK      :     loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c150]] {
// CHECK      :       loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c2]] {
// CHECK      :         loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c3]] {
// CHECK      :           loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c4]] {
// CHECK      :             linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>
// CHECK      :       loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c2]] {
// CHECK      :         loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c3]] {
// CHECK      :           loop.for %{{.*}} = %[[c0]] to %{{.*}} step %[[c4]] {
// CHECK      :             linalg.matmul(%{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>, memref<?x?xf32, #[[STRIDED_2D]]>

