// RUN: dtensor-opt %s -split-input-file -dtensor-reduce-scatter-lowering -verify-diagnostics | FileCheck %s --dump-input=fail

// Check the lowering of DTensorReduceScatter on TPU with sum reduction.
// CHECK-LABEL: func @lower_reduce_scatter_sum_tpu
func.func @lower_reduce_scatter_sum_tpu() -> (tensor<2048x8192xf32>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIMENSION:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[REDUCE_SCATTER_OUT:.*]] = "tf.XlaReduceScatter"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]], %[[SCATTER_DIMENSION]])
  // CHECK-SAME:  reduce_op = "Add"
  // CHECK-NEXT   return %[[REDUCE_SCATTER_OUT]]
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4096x8192xf32>, _layout = ["sharding_specs:unsharded,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"]} : () -> tensor<4096x8192xf32>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xf32>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xf32>
  func.return %3: tensor<2048x8192xf32>
}

// Check the lowering of DTensorReduceScatter on CPU with sum reduction.
// CHECK-LABEL: func @lower_reduce_scatter_sum_cpu
func.func @lower_reduce_scatter_sum_cpu() -> (tensor<2048x8192xf32>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIMENSION:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ALL_REDUCE_OUT:.*]] = "tf.DTensorAllReduce"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Add"
  // CHECK-NEXT:  %[[ALL_SCATTER_OUT:.*]] = "tf.DTensorAllScatter"(%[[ALL_REDUCE_OUT]])
  // CHECK-NEXT   return %[[ALL_SCATTER_OUT]]
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4096x8192xf32>, _layout = ["sharding_specs:unsharded,unsharded, mesh:cpu_mesh|x=2,y=4|*CPU"]} : () -> tensor<4096x8192xf32>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:cpu_mesh|x=2,y=4|*CPU"], device_type = "/job:localhost/replica:0/task:0/device:CPU", reduce_op = "Add"} : (tensor<4096x8192xf32>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xf32>
  func.return %3: tensor<2048x8192xf32>
}

// Check the lowering of DTensorReduceScatter on TPU with any boolean reduction.
// CHECK-LABEL: func @lower_reduce_any_boolean_tpu
func.func @lower_reduce_any_boolean_tpu() -> (tensor<2048x8192xi1>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIMENSION:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[INPUT_CAST:.*]] = "tf.Cast"(%[[CONST_OUT_1]])
  // CHECK-NEXT:  %[[REDUCE_SCATTER_OUT:.*]] = "tf.XlaReduceScatter"(%[[INPUT_CAST]], %[[GROUP_ASSIGNMENT]], %[[SCATTER_DIMENSION]])
  // CHECK-SAME:  reduce_op = "Max"
  // CHECK-NEXT:  %[[OUTPUT_CAST:.*]] = "tf.Cast"(%[[REDUCE_SCATTER_OUT]])
  // CHECK-NEXT   return %[[OUTPUT_CAST]]
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>, _layout = ["sharding_specs:unsharded,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"]} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Any"} : (tensor<4096x8192xi1>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xi1>
  func.return %3: tensor<2048x8192xi1>
}

// Check the lowering of DTensorReduceScatter on CPU with any_boolean reduction.
// CHECK-LABEL: func @lower_reduce_scatter_any_boolean_cpu
func.func @lower_reduce_scatter_any_boolean_cpu() -> (tensor<2048x8192xi1>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIMENSION:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ALL_REDUCE_OUT:.*]] = "tf.DTensorAllReduce"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Any"
  // CHECK-NEXT:  %[[ALL_SCATTER_OUT:.*]] = "tf.DTensorAllScatter"(%[[ALL_REDUCE_OUT]])
  // CHECK-NEXT   return %[[ALL_SCATTER_OUT]]
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>, _layout = ["sharding_specs:unsharded,unsharded, mesh:cpu_mesh|x=2,y=4|*CPU"]} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:cpu_mesh|x=2,y=4|*CPU"], device_type = "/job:localhost/replica:0/task:0/device:CPU", reduce_op = "Any"} : (tensor<4096x8192xi1>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xi1>
  func.return %3: tensor<2048x8192xi1>
}

// Check the lowering of DTensorReduceScatter without input layout.
// CHECK-LABEL: func @lower_reduce_scatter_no_input_layout
func.func @lower_reduce_scatter_no_input_layout() -> (tensor<2048x8192xf32>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[SCATTER_DIMENSION:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ALL_REDUCE_OUT:.*]] = "tf.DTensorAllReduce"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:  reduce_op = "Add"
  // CHECK-NEXT:  %[[ALL_SCATTER_OUT:.*]] = "tf.DTensorAllScatter"(%[[ALL_REDUCE_OUT]])
  // CHECK-NEXT   return %[[ALL_SCATTER_OUT]]
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4096x8192xf32>} : () -> tensor<4096x8192xf32>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:cpu_mesh|x=2,y=4|*CPU"], device_type = "/job:localhost/replica:0/task:0/device:CPU", reduce_op = "Add"} : (tensor<4096x8192xf32>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xf32>
  func.return %3: tensor<2048x8192xf32>
}

// -----

// Check for error of DTensorReduceScatter on TPU with sum boolean reduction.
func.func @lower_reduce_sum_boolean_tpu() -> (tensor<2048x8192xi1>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<4096x8192xi1>, _layout = ["sharding_specs:unsharded,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"]} : () -> tensor<4096x8192xi1>
  %1 = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{reduce for boolean only supports 'All' or 'Any' reduction}}
  %3 = "tf.DTensorReduceScatter"(%0, %1, %2) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=4|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xi1>, tensor<4x2xi32>, tensor<i32>) -> tensor<2048x8192xi1>
  func.return %3: tensor<2048x8192xi1>
}
