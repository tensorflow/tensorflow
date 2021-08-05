// RUN: tf-tfrt-opt -split-input-file -tf-cpurt-pipeline %s | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @tanh_lower_and_fuse
// CHECK-SAME: %[[ARG:.*]]: memref<?x32xf32>
func @tanh_lower_and_fuse(%arg0: tensor<?x32xf32>) -> tensor<?x32xf32> {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[ARG]], %[[C0]]
  // CHECK: %[[MEMREF:.*]] = memref.alloc(%[[DIM]]) : memref<?x32xf32>

  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG]] : memref<?x32xf32>)
  // CHECK-SAME: outs(%[[MEMREF]] : memref<?x32xf32>)
  // CHECK: tanh
  // CHECK-NEXT: tanh

  // CHECK: return %[[MEMREF]]
  %0 = "tf.Tanh"(%arg0): (tensor<?x32xf32>) -> tensor<?x32xf32>
  %1 = "tf.Tanh"(%0): (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1 : tensor<?x32xf32>
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @sigmoid_dynamic_dim
func @sigmoid_dynamic_dim(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  %0 = "tf.Sigmoid"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK: #map0 = affine_map<(d0) -> ()>
// CHECK: #map1 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @add_scalar_with_vec
func @add_scalar_with_vec(%arg0: tensor<f32>,
                          %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: linalg.generic
  %0 = "tf.AddV2"(%arg0, %arg1): (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @tf_binary_with_bcast
func @tf_binary_with_bcast(%arg0: tensor<?x1xf32>,
                           %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK-NOT: shape.
  // CHECK: %[[LHS:.*]] = memref.reinterpret_cast
  // CHECK: %[[RHS:.*]] = memref.reinterpret_cast
  // CHECK: linalg.generic {{.*}} ins(%[[LHS]], %[[RHS]] :
  // CHECK:   mulf
  %0 = "tf.Mul"(%arg0, %arg1)
       : (tensor<?x1xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tf_binary_with_bcast_and_fusion
// CHECK-SAME: %[[ARG0:.*]]: memref<?x4xf32>,
// CHECK-SAME: %[[ARG1:.*]]: memref<4xf32>,
// CHECK-SAME: %[[ARG2:.*]]: memref<4xf32>
func @tf_binary_with_bcast_and_fusion(%arg0: tensor<?x4xf32>,
                                      %arg1: tensor<4xf32>,
                                      %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  // CHECK:      linalg.generic
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] : {{.*}})
  // CHECK:        math.log1p
  // CHECK-NEXT:   subf
  // CHECK-NEXT:   mulf
  // CHECK-NEXT:   linalg.yield
  // CHECK-NOT: linalg.generic
  %0 = "tf.Log1p"(%arg0)
       : (tensor<?x4xf32>) -> tensor<?x4xf32>
  %1 = "tf.Sub"(%0, %arg1)
       : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  %2 = "tf.Mul"(%1, %arg2)
       : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  return %2 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tf_lower_matmul
// CHECK-SAME: %[[ARG0:.*]]: memref<?x?xf32>,
// CHECK-SAME: %[[ARG1:.*]]: memref<?x?xf32>
func @tf_lower_matmul(%arg0: tensor<?x?xf32>,
                      %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-NOT: linalg.copy
  // CHECK: %[[DIM_M:.*]] = memref.dim %[[ARG0]], %c0 : memref<?x?xf32>
  // CHECK: %[[DIM_N:.*]] = memref.dim %[[ARG1]], %c1 : memref<?x?xf32>
  // CHECK-NOT: linalg.copy
  // Tiling for register reuse.
  // CHECK: scf.for %[[M:.*]] = %c0 to %[[DIM_M]] step %c[[MR:[0-9]+]]
  // CHECK: scf.for %[[N:.*]] = %c0 to %[[DIM_N]] step %c[[NR:[0-9]+]]
  // Unrolled tile matmul in vector dialect goes here. It is too large to match.
  // CHECK: scf.yield %[[TILE:.*]] : vector<[[MR]]x[[NR]]xf32>
  %0 = "tf.MatMul"(%arg0, %arg1) { transpose_a = false, transpose_b = false}
       : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @cast_sub
func @cast_sub(%arg0: tensor<?x32xi16>, %arg1: tensor<?x?x32xf16>)
    -> tensor<?x?x32xf16> {
  // CHECK:      linalg.generic
  // CHECK-SAME: outs(%[[RESULT_BUF:.*]] : memref<?x?x32xf16>)
  // CHECK-SAME: {
  // CHECK:      ^bb0(%[[LHS:.*]]: f16, %[[RHS:.*]]: i16, %{{.*}}: f16):
  // CHECK:        %[[RHS_CASTED:.*]] = sitofp %[[RHS]] : i16 to f16
  // CHECK:        %[[RESULT:.*]] = subf %[[LHS]], %[[RHS_CASTED]] : f16
  // CHECK:        linalg.yield %[[RESULT]] : f16
  // CHECK:      }
  // CHECK:      return %[[RESULT_BUF]] : memref<?x?x32xf16>
  %0 = "tf.Cast"(%arg0) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = "tf.Sub"(%arg1, %0) : (tensor<?x?x32xf16>, tensor<?x32xf16>)
      -> tensor<?x?x32xf16>
  return %1 : tensor<?x?x32xf16>
}

// -----

// CHECK: #map0 = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @tf_transpose_const_perm
func @tf_transpose_const_perm(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<3x2xf32>
  // CHECK: linalg.generic {indexing_maps = [#map0, #map1]
  // CHECK-SAME: ins(%arg0 : memref<2x3xf32>)
  // CHECK-SAME: outs(%[[OUT]] : memref<3x2xf32>)
  %0 = "tf.Const"() { value = dense<[1, 0]> : tensor<2xi32> }
         : () -> tensor<2xi32>
  %1 = "tf.Transpose"(%arg0, %0)
         : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @tf_transpose_after_transpose
func @tf_transpose_after_transpose(%arg0: tensor<?x?x?xf32>)
                                  -> tensor<?x?x?xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloc
  // CHECK: linalg.generic {indexing_maps = [#map0, #map1]
  // CHECK-SAME: ins(%arg0 :  memref<?x?x?xf32>)
  // CHECK-SAME: outs(%[[OUT]] :  memref<?x?x?xf32>)
  // CHECK-NOT: linalg.generic
  %0 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi32> }
         : () -> tensor<3xi32>
  %1 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi32> }
         : () -> tensor<3xi32>
  %2 = "tf.Transpose"(%arg0, %0)
         : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %3 = "tf.Transpose"(%2, %1)
         : (tensor<?x?x?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %3 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: @bias_add_and_relu
// CHECK-SAME: %[[ARG0:.*]]: memref<?x32xf32>
// CHECK-SAME: %[[ARG1:.*]]: memref<32xf32>
func @bias_add_and_relu(%arg0: tensor<?x32xf32>,
                        %arg1: tensor<32xf32>) -> tensor<?x32xf32> {
  // CHECK:      linalg.generic
  // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : {{.*}})
  // CHECK:        addf
  // CHECK:        cmpf ogt
  // CHECK:        select
  // CHECK:        cmpf uno
  // CHECK:        select
  // CHECK-NEXT:   linalg.yield
  // CHECK-NOT:  linalg.generic
  %0 = "tf.BiasAdd"(%arg0, %arg1)
         : (tensor<?x32xf32>, tensor<32xf32>) -> tensor<?x32xf32>
  %1 = "tf.Relu"(%0): (tensor<?x32xf32>) -> tensor<?x32xf32>
  return %1 : tensor<?x32xf32>
}

// -----

// CHECK-LABEL: @sub_sub
func @sub_sub(%arg0: tensor<?x32xf16>, %arg1: tensor<?x32xf16>, %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK:      linalg.generic
  // CHECK-SAME: outs(%[[RESULT_BUF:.*]] : memref<?x?x32xf16>)
  // CHECK:      ^bb0(%[[A:.*]]: f16, %[[B:.*]]: f16, %[[C:.*]]: f16, %{{.*}}: f16):
  // CHECK:        %[[TMP:.*]] = subf %[[B]], %[[C]]
  // CHECK:        %[[RESULT:.*]] = subf %[[A]], %[[TMP]]
  // CHECK:        linalg.yield %[[RESULT]]
  // CHECK:      return %[[RESULT_BUF]] : memref<?x?x32xf16>
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<?x32xf16>, tensor<?x32xf16>) -> tensor<?x32xf16>
  %1 = "tf.Sub"(%arg2, %0) : (tensor<?x?x32xf16>, tensor<?x32xf16>) -> tensor<?x?x32xf16>
  return %1 : tensor<?x?x32xf16>
}

// -----

// CHECK-LABEL: @strided_slice_1d_to_0d
func @strided_slice_1d_to_0d(%arg0: tensor<3xi32>) -> tensor<i32> {
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK:      %[[ALLOC:.*]] = memref.alloc() : memref<1xi32>
  // CHECK:      %[[SUBVIEW:.*]] = memref.subview %arg0[0] [1] [1]
  // CHECK-SAME:                 : memref<3xi32> to memref<1xi32>
  // CHECK:      linalg.copy(%[[SUBVIEW]], %[[ALLOC]])
  // CHECK:      %[[RET:.*]] = memref.collapse_shape %[[ALLOC]]
  // CHECK:      return %[[RET]]
  %0 = "tf.StridedSlice"(%arg0, %cst_1, %cst_0, %cst_0)
       {
         begin_mask       = 0 : i64,
         ellipsis_mask    = 0 : i64,
         end_mask         = 0 : i64,
         new_axis_mask    = 0 : i64,
         shrink_axis_mask = 1 : i64
       } : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
         -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK: memref.global "private" constant @__constant_2xi32 : memref<2xi32> = dense<[0, 1]>
// CHECK-LABEL: @constant_folding
func @constant_folding() -> tensor<2xi32> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[CONST:.*]] = memref.get_global @__constant_2xi32 : memref<2xi32>
  // CHECK: return %[[CONST]]
  %2 = "tf.Pack"(%0, %1) {axis = 0 : i64}
       : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  return %2 : tensor<2xi32>
}
