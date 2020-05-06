// RUN: tf-opt %s -split-input-file -tf-annotate-parameter-replication | FileCheck %s --dump-input=fail

// Tests that an operand from outside the replicated region is annotated.

module attributes {tf.versions = {producer = 888 : i32}} {
  // CHECK-LABEL: func @annotate_broadcast_values
  func @annotate_broadcast_values(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf._A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf._B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %5:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      %2 = "tf._F"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
      %3 = "tf.Identity"(%1) : (tensor<?xi32>) -> tensor<?xi32>
      %4 = "tf_device.cluster_func"(%ri_0, %3, %2) {func = @_func, device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
      tf_device.return %4 : tensor<?xi32>
    }
    %6 = "tf._C"(%5#1) : (tensor<?xi32>) -> tensor<?xi32>
    return %6 : tensor<?xi32>
  }

  // CHECK-LABEL: func @_func
  // CHECK-SAME: %[[ARG0:.*]]: tensor<?xi32>,
  // CHECK-SAME: %[[ARG1:.*]]: tensor<?xi32> {tf_device.is_same_data_across_replicas = true}
  // CHECK-SAME: %[[ARG2:.*]]: tensor<?xi32>)
  func @_func(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf._D"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests that a mirrored variable parameter is annotated.

module attributes {tf.versions = {producer = 888 : i32}} {
  // CHECK-LABEL: func @annotate_mirrored_variable
  func @annotate_mirrored_variable(
    %arg0: tensor<!tf.resource<tensor<?xi32>>>,
    %arg1: tensor<!tf.resource<tensor<?xi32>>>,
    %arg2: tensor<!tf.resource<tensor<?xi32>>>,
    %arg3: tensor<!tf.resource<tensor<?xi32>>>,
    %arg4: tensor<!tf.resource<tensor<?xi32>>>,
    %arg5: tensor<!tf.resource<tensor<?xi32>>>) -> tensor<?xi32> {
    %3:2 = tf_device.replicate(
      [%arg0, %arg1] as %ri_0: tensor<!tf.resource<tensor<?xi32>>>,
      [%arg2, %arg3] as %ri_1: tensor<!tf.resource<tensor<?xi32>>>,
      [%arg4, %arg5] as %ri_2: tensor<!tf.resource<tensor<?xi32>>>) {_mirrored_variable_indices = [0, 2], n = 2 : i32} {
      %0 = "tf.ReadVariableOp"(%ri_0): (tensor<!tf.resource<tensor<?xi32>>>) -> tensor<?xi32>
      %1 = "tf.ReadVariableOp"(%ri_1): (tensor<!tf.resource<tensor<?xi32>>>) -> tensor<?xi32>
      %2 = "tf_device.cluster_func"(%0, %1, %ri_2) {func = @_func, device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<!tf.resource<tensor<?xi32>>>) -> tensor<?xi32>
      tf_device.return %2 : tensor<?xi32>
    }
    %4 = "tf._C"(%3#1) : (tensor<?xi32>) -> tensor<?xi32>
    return %4 : tensor<?xi32>
  }

  // CHECK-LABEL: func @_func
  // CHECK-SAME: %[[ARG0:.*]]: tensor<?xi32> {tf_device.is_same_data_across_replicas = true},
  // CHECK-SAME: %[[ARG1:.*]]: tensor<?xi32>,
  // CHECK-SAME: %[[ARG2:.*]]: tensor<!tf.resource<tensor<?xi32>>> {tf_device.is_same_data_across_replicas = true}
  func @_func(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<!tf.resource<tensor<?xi32>>>) -> tensor<?xi32> {
    %0 = "tf._D"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests that a non-replicated ClusterFuncOp is not annotated.

module attributes {tf.versions = {producer = 888 : i32}} {
  // CHECK-LABEL: func @do_not_annotate_without_replicate
  func @do_not_annotate_without_replicate(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf._A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf._B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf_device.cluster_func"(%0, %1) {func = @_func, device = ""} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf._C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    return %3 : tensor<?xi32>
  }

  // CHECK-LABEL: func @_func
  // CHECK-NOT: tf_device.is_same_data_across_replicas
  func @_func(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf._D"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}
