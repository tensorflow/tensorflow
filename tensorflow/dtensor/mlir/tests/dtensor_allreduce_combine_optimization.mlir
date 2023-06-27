// RUN: dtensor-opt %s -split-input-file -dtensor-allreduce-combine-optimization -verify-diagnostics | FileCheck %s

// Check that independent DTensorAllReduce ops of the same element type and group assignment are combined.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:      %[[VAL_1:.*]] = "tf.Const"
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  // CHECK:      %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"()
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // CHECK:      %[[VAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  //
  // CHECK:      %[[FILL:.*]] = "tf.Fill"
  // CHECK:      %[[FLATTEN_1:.*]] = "tf.Reshape"(%[[VAL_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_1:.*]] = "tf.TensorStridedSliceUpdate"(%[[FILL]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_1]])
  // CHECK:      %[[FLATTEN_2:.*]] = "tf.Reshape"(%[[VAL_2]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_2:.*]] = "tf.TensorStridedSliceUpdate"(%[[UPDATE_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_2]])
  // CHECK:      %[[ALL_REDUCE:.*]] = "tf.DTensorAllReduce"(%[[UPDATE_2]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
  // CHECK:      %[[SLICE_1:.*]] = "tf.Slice"(%[[ALL_REDUCE]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_1:.*]] = "tf.Reshape"(%[[SLICE_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[SLICE_2:.*]] = "tf.Slice"(%[[ALL_REDUCE]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_2:.*]] = "tf.Reshape"(%[[SLICE_2]], %cst_{{[0-9]*}})
  //
  // CHECK:      %[[ADD:.*]] = "tf.Add"(%[[UNFLATTEN_1]], %[[UNFLATTEN_2]])
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %5 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %6 = "tf.DTensorAllReduce"(%4, %5) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %7 = "tf.Add"(%3, %6) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    "tf_device.return"(%7) : (tensor<4x4xf32>) -> ()
  }) : () -> tensor<4x4xf32>
  "func.return"() : () -> ()
}

// -----
// Check that two groups of interdependent DTensorAllReduce ops are combined layer by layer.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:      %[[VAL_1:.*]] = "tf.Const"
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  // CHECK:      %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"()
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // CHECK:      %[[VAL_2:.*]] = "tf.Const"
  // CHECK-SAME:   {value = dense<{{1.0.*}}> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  //
  //
  // CHECK:      %[[ALL_REDUCE_0:.*]] = "tf.DTensorAllReduce"(%[[VAL_2]], %[[GROUP_ASSIGNMENT]])
  //
  // CHECK:      %[[FILL_1:.*]] = "tf.Fill"
  // CHECK:      %[[FLATTEN_1:.*]] = "tf.Reshape"(%[[VAL_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_1:.*]] = "tf.TensorStridedSliceUpdate"(%[[FILL_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_1]])
  // CHECK:      %[[FLATTEN_2:.*]] = "tf.Reshape"(%[[ALL_REDUCE_0]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_2:.*]] = "tf.TensorStridedSliceUpdate"(%[[UPDATE_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_2]])
  // CHECK:      %[[ALL_REDUCE_1:.*]] = "tf.DTensorAllReduce"(%[[UPDATE_2]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
  // CHECK:      %[[SLICE_1:.*]] = "tf.Slice"(%[[ALL_REDUCE_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_1:.*]] = "tf.Reshape"(%[[SLICE_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[SLICE_2:.*]] = "tf.Slice"(%[[ALL_REDUCE_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_2:.*]] = "tf.Reshape"(%[[SLICE_2]], %cst_{{[0-9]*}})
  //
  // CHECK:      %[[FILL_2:.*]] = "tf.Fill"
  // CHECK:      %[[FLATTEN_3:.*]] = "tf.Reshape"(%[[UNFLATTEN_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_3:.*]] = "tf.TensorStridedSliceUpdate"(%[[FILL_2]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_3]])
  // CHECK:      %[[FLATTEN_4:.*]] = "tf.Reshape"(%[[UNFLATTEN_2]], %cst_{{[0-9]*}})
  // CHECK:      %[[UPDATE_4:.*]] = "tf.TensorStridedSliceUpdate"(%[[UPDATE_3]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[FLATTEN_4]])
  // CHECK:      %[[ALL_REDUCE_2:.*]] = "tf.DTensorAllReduce"(%[[UPDATE_4]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
  // CHECK:      %[[SLICE_3:.*]] = "tf.Slice"(%[[ALL_REDUCE_2]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_3:.*]] = "tf.Reshape"(%[[SLICE_3]], %cst_{{[0-9]*}})
  // CHECK:      %[[SLICE_4:.*]] = "tf.Slice"(%[[ALL_REDUCE_2]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[UNFLATTEN_4:.*]] = "tf.Reshape"(%[[SLICE_4]], %cst_{{[0-9]*}})
  //
  // CHECK:      %[[ADD:.*]] = "tf.Add"(%[[UNFLATTEN_3]], %[[UNFLATTEN_4]])
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.DTensorAllReduce"(%3, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %5 = "tf.Const"() {value = dense<1.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %6 = "tf.DTensorAllReduce"(%5, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %7 = "tf.DTensorAllReduce"(%6, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %8 = "tf.DTensorAllReduce"(%7, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %9 = "tf.Add"(%4, %8) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

    "tf_device.return"(%9) : (tensor<4x4xf32>) -> ()
  }) : () -> tensor<4x4xf32>
  "func.return"() : () -> ()
}

// -----

// Check that DTensorAllReduce ops across region boundaries are not combined.
// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:      %[[VAL:.*]] = "tf.Const"
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  // CHECK:      %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"()
  // CHECK-SAME:   {value = dense<{{.*}}> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // CHECK:      %[[ALL_REDUCE_1:.*]] = "tf.DTensorAllReduce"(%[[VAL]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
  //
  // CHECK:      "tf.WhileRegion"
  //
  // CHECK:      "tf.ToBool"
  // CHECK-NEXT: "tf.Yield"
  //
  // CHECK:      %[[WHILE_FILL:.*]] = "tf.Fill"
  // CHECK:      %[[WHILE_FLATTEN_1:.*]] = "tf.Reshape"(%[[VAL]], %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_UPDATE_1:.*]] = "tf.TensorStridedSliceUpdate"(%[[WHILE_FILL]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[WHILE_FLATTEN_1]])
  // CHECK:      %[[WHILE_FLATTEN_2:.*]] = "tf.Reshape"(%[[VAL]], %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_UPDATE_2:.*]] = "tf.TensorStridedSliceUpdate"(%[[WHILE_UPDATE_1]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %cst_{{[0-9]*}}, %[[WHILE_FLATTEN_2]])
  // CHECK:      %[[WHILE_ALL_REDUCE:.*]] = "tf.DTensorAllReduce"(%[[WHILE_UPDATE_2]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
  // CHECK:      %[[WHILE_SLICE_1:.*]] = "tf.Slice"(%[[WHILE_ALL_REDUCE]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_UNFLATTEN_1:.*]] = "tf.Reshape"(%[[WHILE_SLICE_1]], %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_SLICE_2:.*]] = "tf.Slice"(%[[WHILE_ALL_REDUCE]], %cst_{{[0-9]*}}, %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_UNFLATTEN_2:.*]] = "tf.Reshape"(%[[WHILE_SLICE_2]], %cst_{{[0-9]*}})
  // CHECK:      %[[WHILE_ADD:.*]] = "tf.Add"(%[[WHILE_UNFLATTEN_1]], %[[WHILE_UNFLATTEN_2]])
  // CHECK:      "tf.Yield"(%[[WHILE_ADD]])
  //
  // CHECK:      %[[ALL_REDUCE_2:.*]] = "tf.DTensorAllReduce"(%[[VAL]], %[[GROUP_ASSIGNMENT]])
  // CHECK-SAME:   (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
  // CHECK:      %[[ADD:.*]] = "tf.Add"(%[[ALL_REDUCE_1]], %[[ALL_REDUCE_2]])
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
    %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %3 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %4 = "tf.WhileRegion"(%1) ({
    ^bb0(%arg: tensor<4x4xf32>):
      %5 = "tf.ToBool"(%arg) : (tensor<4x4xf32>) -> tensor<i1>
      "tf.Yield"(%5) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg: tensor<4x4xf32>):
      %5 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %6 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %7 = "tf.Add"(%5, %6) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      "tf.Yield"(%7) : (tensor<4x4xf32>) -> ()
    }) {is_stateless = true} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    %6 = "tf.Add"(%3, %5) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    "tf_device.return"(%6) : (tensor<4x4xf32>) -> ()
  }) : () -> tensor<4x4xf32>
  "func.return"() : () -> ()
}

// -----
module attributes {dtensor.all_reduce_combiner.num_ops_in_group = 2} {
  // Check that when DTENSOR_ALLREDUCE_COMBINE_OPTIMIZATION_GROUP_SIZE is set, 
  // independent DTensorAllReduce ops of the same element type and group
  // assignment are combined no more than the specified size. Use of dummy All-
  // Reduces (of the same input) gaurantees ops to be grouped together if envvar
  // is not specified.
  // The following scenario should have 3 groups *without* envvar set:
  // group 1: 2 all reduces
  // group 2: 3 all reduces
  // group 3: 4 all reduces
  // With DTENSOR_ALLREDUCE_COMBINE_OPTIMIZATION_GROUP_SIZE=2, we expect to have
  // the following 5 groups:
  // group 1: 2 all reduces (original group, test for exact match of size)
  // group 2: 2 all reduces (2/3 of original group 2, test of uneven split)
  // group 3: 1 all reduces (1/3 of original group 2, test of uneven split)
  // group 4: 2 all reduces (2/4 of original group 3, test of even split)
  // group 5: 2 all reduces (2/4 of original group 3, test of even split)
  // CHECK-LABEL: func @main
  func.func @main() {
    // CHECK:      %[[ALL_REDUCE_1:.*]] = "tf.DTensorAllReduce"
    // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
    // CHECK:      %[[ALL_REDUCE_2:.*]] = "tf.DTensorAllReduce"
    // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
    // CHECK:      %[[ALL_REDUCE_3:.*]] = "tf.DTensorAllReduce"
    // CHECK-SAME:   (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
    // CHECK:      %[[ALL_REDUCE_4:.*]] = "tf.DTensorAllReduce"
    // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
    // CHECK:      %[[ALL_REDUCE_5:.*]] = "tf.DTensorAllReduce"
    // CHECK-SAME:   (tensor<1024xf32>, tensor<2x2xi32>) -> tensor<1024xf32>
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
      %2 = "tf.Const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %3 = "tf.Const"() {value = dense<1.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
      %4 = "tf.Const"() {value = dense<[[3, 2], [1, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %5 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %6 = "tf.DTensorAllReduce"(%1, %2) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %7 = "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %8 = "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %9 = "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %10 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
      %11 = "tf.Const"() {value = dense<[[0, 1], [3, 2]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %12 = "tf.DTensorAllReduce"(%10, %11) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %13 = "tf.DTensorAllReduce"(%10, %11) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %14 = "tf.DTensorAllReduce"(%10, %11) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %15 = "tf.DTensorAllReduce"(%10, %11) {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*GPU"], device_type = "GPU", reduce_op = "Add"} : (tensor<4x4xf32>, tensor<2x2xi32>) -> tensor<4x4xf32>
      %16 = "tf.Add"(%9, %15) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      "tf_device.return"(%16) : (tensor<4x4xf32>) -> ()
    }) : () -> tensor<4x4xf32>
    "func.return"() : () -> ()
  }
}