// RUN: tf-opt %s -split-input-file -tf-device-cluster-outlining | FileCheck %s

// Tests simple case of a single `tf_device.cluster`.

// CHECK-LABEL: func @single_cluster
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @single_cluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[CLUSTER_OUTPUT:[0-9]*]] = "tf_device.cluster_func"(%[[A_OUTPUT]]) {func = @[[CLUSTER:.*]]}
      %3 = "tf_device.cluster"() ( {
        %4 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) {} : () -> tensor<?xi32>

      // CHECK: tf_executor.yield %[[CLUSTER_OUTPUT]]
      tf_executor.yield %3 : tensor<?xi32>
    }
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[CLUSTER]]
// CHECK-SAME: (%[[CLUSTER_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[CLUSTER_ARG_0]])
// CHECK: return %[[B_OUTPUT]]

// -----

// Tests that multiple `tf_device.cluster` that depend on each other are
// correctly handled.

// CHECK-LABEL: func @multiple_clusters
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @multiple_clusters(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
      %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[CLUSTER_0_OUTPUT:[0-9]*]] = "tf_device.cluster_func"(%[[A_OUTPUT]]) {func = @[[CLUSTER_0:.*]]}
      %3 = "tf_device.cluster"() ( {
        %6 = "tf.B"(%2) : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %6 : tensor<?xi32>
      }) {} : () -> tensor<?xi32>

      // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[CLUSTER_0_OUTPUT]])
      %4 = "tf.D"(%3) : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: %[[CLUSTER_1_OUTPUT:[0-9]*]] = "tf_device.cluster_func"(%[[CLUSTER_0_OUTPUT]], %[[D_OUTPUT]]) {func = @[[CLUSTER_1:.*]]}
      %5 = "tf_device.cluster"() ( {
        %6 = "tf.E"(%3) : (tensor<?xi32>) -> tensor<?xi32>
        %7 = "tf.F"(%4, %6) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %7 : tensor<?xi32>
      }) {} : () -> tensor<?xi32>

      // CHECK: tf_executor.yield %[[CLUSTER_1_OUTPUT]]
      tf_executor.yield %5 : tensor<?xi32>
    }
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[CLUSTER_0]]
// CHECK-SAME: (%[[CLUSTER_0_ARG_0:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[CLUSTER_0_ARG_0]])
// CHECK: return %[[B_OUTPUT]]

// CHECK: func private @[[CLUSTER_1]]
// CHECK-SAME: (%[[CLUSTER_1_ARG_0:[a-z0-9]*]]: tensor<?xi32>, %[[CLUSTER_1_ARG_1:[a-z0-9]*]]: tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[CLUSTER_1_ARG_0]])
// CHECK: %[[F_OUTPUT:[0-9]*]] = "tf.F"(%[[CLUSTER_1_ARG_1]], %[[E_OUTPUT]])
// CHECK: return %[[F_OUTPUT]]

// -----

// Tests outlining clusters with no live-in values.

// CHECK-LABEL: func @cluster_operands
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
func @cluster_operands(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island wraps
      // CHECK: %[[CLUSTER_OUTPUT:[a-z0-9]*]], %{{.*}} = {{.*}} "tf_device.cluster_func"() {func = @[[CLUSTER:.*]]}
      "tf_device.cluster"() ( {
        %3 = "tf.A"() : () -> tensor<?xi32>
        tf_device.return %3 : tensor<?xi32>
      }) {} : () -> tensor<?xi32>
    // CHECK: tf_executor.fetch %[[CLUSTER_OUTPUT]]
    tf_executor.fetch %1#0 : tensor<?xi32>
  }
  return %0 : tensor<?xi32>
}

// CHECK: func private @[[CLUSTER]]
// CHECK-SAME: () -> tensor<?xi32>
// CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"()
// CHECK: return %[[A_OUTPUT]]

// -----

// Tests cluster attributes are copied over to cluster_func.

// CHECK-LABEL: func @cluster_attrs
func @cluster_attrs() -> tensor<?xi32> {
  %0 = "tf_device.cluster"() ( {
    %1 = "tf.A"() : () -> tensor<?xi32>
    tf_device.return %1 : tensor<?xi32>
  }) {cluster_attr = "cluster_attr"} : () -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK: "tf_device.cluster_func"
// CHECK-SAME: cluster_attr = "cluster_attr"
