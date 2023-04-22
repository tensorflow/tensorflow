// RUN: tf-opt %s -tf-drop-while-shape-invariant | FileCheck %s
// RUN: tf-opt %s -tf-drop-while-shape-invariant-in-device-cluster | FileCheck -check-prefix=IN-CLUSTER %s


func @while_cond(%arg0: tensor<*xf32>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  return %0 : tensor<i1>
}

func @while_body(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
  %0 = "tf.SomeOp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// Test that -tf-drop-while-shape-invariant-in-device-cluster pass does not drop
// the shape_invariant attribute from While/WhileRegion ops outside the device
// cluster, while the other pass drops them.

// CHECK-LABEL: while_shape_invariant_outside_cluster
// CHECK-NOT: shape_invariant
// IN-CLUSTER-LABEL: while_shape_invariant_outside_cluster
func @while_shape_invariant_outside_cluster(%arg0: tensor<4xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  // IN-CLUSTER: shape_invariant
  %0 = "tf.While"(%arg0) {cond = @while_cond, body = @while_body, is_stateless = false, shape_invariant} : (tensor<4xf32>) -> (tensor<*xf32>)

  // IN-CLUSTER: shape_invariant
  %1 = "tf.WhileRegion"(%arg0) ( {
  ^cond(%carg0: tensor<*xf32>):
    %2 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.Yield"(%2) : (tensor<i1>) -> ()
  }, {
  ^body(%barg0: tensor<*xf32>):
    %2 = "tf.SomeOp"(%barg0) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false, shape_invariant} : (tensor<4xf32>) -> (tensor<*xf32>)

  return %0, %1 : tensor<*xf32>, tensor<*xf32>
}

// Test that both passes drop the shape_invariant attribute from
// While/WhileRegion ops within a cluster.

// CHECK-LABEL: while_shape_invariant_within_cluster
// CHECK-NOT: shape_invariant
// IN-CLUSTER-LABEL: while_shape_invariant_within_cluster
// IN-CLUSTER-NOT: shape_invariant
func @while_shape_invariant_within_cluster(%arg0: tensor<4xf32>) {
  "tf_device.cluster"() ( {
    %0 = "tf.While"(%arg0) {cond = @while_cond, body = @while_body, is_stateless = false, shape_invariant} : (tensor<4xf32>) -> (tensor<*xf32>)

    %1 = "tf.WhileRegion"(%arg0) ( {
    ^cond(%carg0: tensor<*xf32>):
      %2 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      "tf.Yield"(%2) : (tensor<i1>) -> ()
    }, {
    ^body(%barg0: tensor<*xf32>):
      %2 = "tf.SomeOp"(%barg0) : (tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = false, shape_invariant} : (tensor<4xf32>) -> (tensor<*xf32>)
    tf_device.return
  }) {} : () -> ()

  return
}
