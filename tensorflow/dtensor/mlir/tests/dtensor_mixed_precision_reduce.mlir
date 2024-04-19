// RUN: DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE=4 dtensor-opt -- -split-input-file -dtensor-mixed-precision-reduce -verify-diagnostics %s | FileCheck %s

// Check bfloat16 AllReduce is upcasted for a sufficient group size.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<1x4xbf16>
func.func @main(
  %arg0: tensor<1x4xbf16> {tf._global_shape = #tf_type.shape<8x4>, tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=8|*TPU"})
  -> (tensor<4xbf16> {tf._global_shape = #tf_type.shape<4>}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:  %[[AXIS:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SUM_OUT:.*]] = "tf.Sum"(%[[ARG0]], %[[AXIS]])
  // CHECK-SAME:    -> tensor<4xbf16>
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[UPCAST:.*]] = "tf.Cast"(%[[SUM_OUT]])
  // CHECK-SAME:    (tensor<4xbf16>) -> tensor<4xf32>
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorAllReduce"(%[[UPCAST]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:    -> tensor<4xf32>
  // CHECK-NEXT:  %[[DOWNCAST:.*]] = "tf.Cast"(%[[REDUCTION_OUT]])
  // CHECK-SAME:    _layout = ["sharding_specs:unsharded, mesh:TPU|x=8|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:    (tensor<4xf32>) -> tensor<4xbf16>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<>], _layout = ["sharding_specs: mesh:TPU|x=8|*TPU"], value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Sum"(%arg0, %cst) {_global_shape = [#tf_type.shape<4>], device = "", keep_dims = false} : (tensor<1x4xbf16>, tensor<i32>) -> tensor<4xbf16>
    %cst_0 = "tf.Const"() {value = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>} : () -> tensor<1x8xi32>
    %2 = "tf.DTensorAllReduce"(%1, %cst_0) {_layout = ["sharding_specs:unsharded, mesh:TPU|x=8|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4xbf16>, tensor<1x8xi32>) -> tensor<4xbf16>
    %3 = "tf.Identity"(%2) {_global_shape = [#tf_type.shape<4>], _layout = ["sharding_specs:unsharded, mesh:TPU|x=8|*TPU"], device = ""} : (tensor<4xbf16>) -> tensor<4xbf16>
    tf_device.return %3 : tensor<4xbf16>
  }) {_mesh = "TPU|x=8|*TPU"} : () -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}

// -----

// Check that bfloat16 AllReduce is not upcasted for a small group size.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<1x4xbf16>
func.func @main(
  %arg0: tensor<1x4xbf16> {tf._global_shape = #tf_type.shape<2x4>, tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2|*TPU"})
  -> (tensor<4xbf16> {tf._global_shape = #tf_type.shape<4>}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:  %[[AXIS:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SUM_OUT:.*]] = "tf.Sum"(%[[ARG0]], %[[AXIS]])
  // CHECK-SAME:    -> tensor<4xbf16>
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NOT:   "tf.Cast"
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorAllReduce"(%[[SUM_OUT]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:    -> tensor<4xbf16>
  // CHECK-NOT:   "tf.Cast"
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {_global_shape = [#tf_type.shape<>], _layout = ["sharding_specs: mesh:TPU|x=2|*TPU"], value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Sum"(%arg0, %cst) {_global_shape = [#tf_type.shape<4>], device = "", keep_dims = false} : (tensor<1x4xbf16>, tensor<i32>) -> tensor<4xbf16>
    %cst_0 = "tf.Const"() {value = dense<[[0, 1]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
    %2 = "tf.DTensorAllReduce"(%1, %cst_0) {_layout = ["sharding_specs:unsharded, mesh:TPU|x=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4xbf16>, tensor<1x2xi32>) -> tensor<4xbf16>
    %3 = "tf.Identity"(%2) {_global_shape = [#tf_type.shape<4>], _layout = ["sharding_specs:unsharded, mesh:TPU|x=2|*TPU"], device = ""} : (tensor<4xbf16>) -> tensor<4xbf16>
    tf_device.return %3 : tensor<4xbf16>
  }) {_mesh = "TPU|x=2|*TPU"} : () -> tensor<4xbf16>
  func.return %0 : tensor<4xbf16>
}

// -----

// Check bfloat16 ReduceScatter is upcasted for a sufficient group size.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<512x1024xbf16>
func.func @main(
  %arg0: tensor<512x1024xbf16> {tf._global_shape = #tf_type.shape<4096x1024>, tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=8|*TPU"})
  -> (tensor<512x1024xbf16> {tf._global_shape = #tf_type.shape<4096x1024>}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIM:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[UPCAST:.*]] = "tf.Cast"(%[[ARG0]])
  // CHECK-SAME:    (tensor<512x1024xbf16>) -> tensor<512x1024xf32>
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorReduceScatter"(%[[UPCAST]], %[[GROUP_ASSIGNMENT]], %[[SCATTER_DIM]])
  // CHECK-SAME:    -> tensor<512x1024xf32>
  // CHECK-NEXT:  %[[DOWNCAST:.*]] = "tf.Cast"(%[[REDUCTION_OUT]])
  // CHECK-SAME:    _layout = ["sharding_specs:x,unsharded, mesh:TPU|x=8|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3,/job:localhost/task:0/device:TPU:4,/job:localhost/task:0/device:TPU:5,/job:localhost/task:0/device:TPU:6,/job:localhost/task:0/device:TPU:7"]
  // CHECK-SAME:   (tensor<512x1024xf32>) -> tensor<512x1024xbf16>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi32>} : () -> tensor<1x8xi32>
    %cst_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.DTensorReduceScatter"(%arg0, %cst, %cst_0) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=8|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<512x1024xbf16>, tensor<1x8xi32>, tensor<i32>) -> tensor<512x1024xbf16>
    %3 = "tf.Identity"(%2) {_global_shape = [#tf_type.shape<4096x1024>], _layout = ["sharding_specs:x,unsharded, mesh:TPU|x=8|*TPU"], device = ""} : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    tf_device.return %3 : tensor<512x1024xbf16>
  }) {_mesh = "TPU|x=8|*TPU"} : () -> tensor<512x1024xbf16>
  func.return %0 : tensor<512x1024xbf16>
}

// -----

// Check that bfloat16 ReduceScatter is not upcasted for a small group size.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<512x1024xbf16>
func.func @main(
  %arg0: tensor<512x1024xbf16> {tf._global_shape = #tf_type.shape<1024x1024>, tf._layout = "sharding_specs:x,unsharded, mesh:TPU|x=2|*TPU"})
  -> (tensor<512x1024xbf16> {tf._global_shape = #tf_type.shape<1024x1024>}) {
  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIM:.*]] = "tf.Const"
  // CHECK-NOT:   "tf.Cast"
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorReduceScatter"(%[[ARG0]], %[[GROUP_ASSIGNMENT]], %[[SCATTER_DIM]])
  // CHECK-SAME:    -> tensor<512x1024xbf16>
  // CHECK-NOT:   "tf.Cast"
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {value = dense<[[0, 1]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
    %cst_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.DTensorReduceScatter"(%arg0, %cst, %cst_0) {_layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<512x1024xbf16>, tensor<1x2xi32>, tensor<i32>) -> tensor<512x1024xbf16>
    %3 = "tf.Identity"(%2) {_global_shape = [#tf_type.shape<1024x1024>], _layout = ["sharding_specs:x,unsharded, mesh:TPU|x=2|*TPU"], device = ""} : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    tf_device.return %3 : tensor<512x1024xbf16>
  }) {_mesh = "TPU|x=2|*TPU"} : () -> tensor<512x1024xbf16>
  func.return %0 : tensor<512x1024xbf16>
}
