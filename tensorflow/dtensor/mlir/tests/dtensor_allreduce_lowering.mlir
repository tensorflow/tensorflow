// RUN: dtensor-opt -split-input-file -dtensor-all-reduce-lowering -verify-diagnostics %s| FileCheck %s --dump-input=fail

// Check the lowering of AllReduce on TPU with sum reduction.
// CHECK-LABEL: func @lower_allreduce_sum
func.func @lower_allreduce_sum() -> (tensor<4096x8192xf32>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ALLREDUCE_OUT:.*]] = "tf.XlaAllReduce"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Add"
  // CHECK-NEXT   return %[[ALLREDUCE_OUT]]
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4096x8192xf32>} : () -> tensor<4096x8192xf32>
  %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xf32>, tensor<4x2xi32>) -> tensor<4096x8192xf32>
  func.return %2: tensor<4096x8192xf32>
}

// -----

// Check the lowering of AllReduce on TPU with any boolean reduction.
// CHECK-LABEL: func @lower_allreduce_any_boolean
func.func @lower_allreduce_any_boolean() -> (tensor<4096x8192xi1>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[INPUT_CAST:.*]] = "tf.Cast"(%[[CONST_OUT_1]])
  // CHECK-NEXT:  %[[ALLREDUCE_OUT:.*]] = "tf.XlaAllReduce"(%[[INPUT_CAST]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Max"
  // CHECK-NEXT:  %[[OUTPUT_CAST:.*]] = "tf.Cast"(%[[ALLREDUCE_OUT]])
  // CHECK-NEXT   return %[[OUTPUT_CAST]]
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Any"} : (tensor<4096x8192xi1>, tensor<4x2xi32>) -> tensor<4096x8192xi1>
  func.return %2: tensor<4096x8192xi1>
}

// -----

// Check the lowering of AllReduce on TPU with all boolean reduction.
// CHECK-LABEL: func @lower_allreduce_all_boolean
func.func @lower_allreduce_all_boolean() -> (tensor<4096x8192xi1>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[INPUT_CAST:.*]] = "tf.Cast"(%[[CONST_OUT_1]])
  // CHECK-NEXT:  %[[ALLREDUCE_OUT:.*]] = "tf.XlaAllReduce"(%[[INPUT_CAST]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Min"
  // CHECK-NEXT:  %[[OUTPUT_CAST:.*]] = "tf.Cast"(%[[ALLREDUCE_OUT]])
  // CHECK-NEXT   return %[[OUTPUT_CAST]]
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "All"} : (tensor<4096x8192xi1>, tensor<4x2xi32>) -> tensor<4096x8192xi1>
  func.return %2: tensor<4096x8192xi1>
}

// -----

// Check for error  of AllReduce on TPU with all boolean reduction.
func.func @lower_allreduce_sum_boolean() -> (tensor<4096x8192xi1>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // expected-error @+1 {{reduce for boolean only supports 'All'/'Min' or 'Any'/'Max' reduction}}
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xi1>, tensor<4x2xi32>) -> tensor<4096x8192xi1>
  func.return %2: tensor<4096x8192xi1>
}

// -----

// CHECK-LABEL: func @lower_all_reduce_gpu_mesh
func.func @lower_all_reduce_gpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<4096x8192xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:gpu_mesh|x=2,y=4|*GPU"}) -> tensor<4096x8192xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:       %[[DEVICE_ID_RESHAPE:.*]] = "tf.Reshape"(%arg0
  // CHECK:       %[[RELATIVE_DEVICE_ID:.*]] = "tf.Sub"(%[[DEVICE_ID_RESHAPE]]
  // CHECK-DAG:   %[[CONST_1:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>}
  // CHECK-DAG:   %[[DEVICE_ID_TO_GROUP_KEY:.*]] = "tf.Const"() {value = dense<[[[GROUP_KEYS:.*]]]> : tensor<8xi32>}
  // CHECK:       %[[GROUP_KEY_SLICE:.*]] = "tf.Slice"(%[[DEVICE_ID_TO_GROUP_KEY]], %[[RELATIVE_DEVICE_ID]], %[[CONST_1]]
  // CHECK:       %[[GROUP_KEY_RESHAPE:.*]] = "tf.Reshape"(%[[GROUP_KEY_SLICE]]
  // CHECK-DAG:   %[[INSTANCE_KEY:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} 
  // CHECK-DAG:   %[[GROUP_SIZE:.*]] = "tf.Const"() {value = dense<2> : tensor<i32>}
  // CHECK:       %[[REDUCE_OUT:.*]] = "tf.CollectiveReduceV2"(%arg1, %[[GROUP_SIZE]], %[[GROUP_KEY_RESHAPE]], %[[INSTANCE_KEY]])
  // CHECK-SAME:  final_op = "Id"
  // CHECK-SAME:  merge_op = "Add"
  // CHECK-NEXT   tf_device.return %[[REDUCE_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %3 = "tf.DTensorAllReduce"(%arg1, %1) {_layout = ["sharding_specs:x,unsharded, mesh:gpu_mesh|x=2,y=4|*GPU"], device_type = "/job:localhost/replica:0/task:0/device:GPU", reduce_op = "Add"} : (tensor<4096x8192xf32>, tensor<4x2xi32>) -> tensor<4096x8192xf32>
    tf_device.return %3 : tensor<4096x8192xf32>
  }) {_mesh = "GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> tensor<4096x8192xf32>
  func.return %0 : tensor<4096x8192xf32>
}

// -----

// Tests unsupported integer types are promoted to i64.
// CHECK-LABEL: func @lower_all_reduce_i8_gpu_mesh
func.func @lower_all_reduce_i8_gpu_mesh(%arg0: tensor<i32>,
           %arg1: tensor<4096x8192xi8> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:gpu_mesh|x=2,y=4|*GPU"}) -> tensor<4096x8192xi8> {
  // CHECK:      "tf_device.cluster"
  // CHECK:       %[[PRECAST_OUT:.*]] = "tf.Cast"(%arg1)
  // CHECK-SAME:      -> tensor<4096x8192xi64>
  // CHECK:       %[[REDUCE_OUT:.*]] = "tf.CollectiveReduceV2"(%[[PRECAST_OUT]],
  // CHECK:       %[[POSTCAST_OUT:.*]] = "tf.Cast"(%[[REDUCE_OUT]])
  // CHECK-SAME:      -> tensor<4096x8192xi8>
  // CHECK-NEXT   tf_device.return %[[POSTCAST_OUT]]
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %3 = "tf.DTensorAllReduce"(%arg1, %1) {_layout = ["sharding_specs:x,unsharded, mesh:gpu_mesh|x=2,y=4|*GPU"], device_type = "/job:localhost/replica:0/task:0/device:GPU", reduce_op = "Add"} : (tensor<4096x8192xi8>, tensor<4x2xi32>) -> tensor<4096x8192xi8>
    tf_device.return %3 : tensor<4096x8192xi8>
  }) {_mesh = "GPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:GPU:0,/job:localhost/task:0/device:GPU:1,/job:localhost/task:0/device:GPU:2,/job:localhost/task:0/device:GPU:3"} : () -> tensor<4096x8192xi8>
  func.return %0 : tensor<4096x8192xi8>
}

