// RUN: dtensor-opt %s -split-input-file -dtensor-all-reduce-lowering -verify-diagnostics| FileCheck %s --dump-input=fail

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
  // expected-error @+1 {{reduce for boolean only supports 'All' or 'Any' reduction}}
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xi1>, tensor<4x2xi32>) -> tensor<4096x8192xi1>
  func.return %2: tensor<4096x8192xi1>
}
