// RUN: tf-opt %s -split-input-file -tf-device-cluster-formation | FileCheck %s -dump-input-on-failure

// Simple case, single device cluster.

module {
  // CHECK-LABEL: func @singlecluster
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @singlecluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
    %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %4 = "tf.C"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

    // CHECK: tf_device.return %[[C_OUTPUT]]
    // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

    // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[TPU0_OUTPUT]])
    %5 = "tf.D"(%4) : (tensor<?xi32>) -> tensor<?xi32>
    return %5 : tensor<?xi32>
  }
}

// -----

// Simple case, single device cluster, nested in a tf_executor.graph

module {
  // CHECK-LABEL: func @singlecluster
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @singlecluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %4 = "tf.C"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[C_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[TPU0_OUTPUT]])
        %5 = "tf.D"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_executor.yield %5 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Single device cluster, live-in value comes directly from function argument.

module {
  // CHECK-LABEL: func @arglivein
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @arglivein(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.A"(%arg0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]], %[[ARG_0]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %4 = "tf.B"(%3, %arg0) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[B_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[TPU0_OUTPUT]])
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_executor.yield %5 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Single device cluster, live-in value comes from other islands.

module {
  // CHECK-LABEL: func @argliveinotherislands
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @argliveinotherislands(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      // CHECK: %[[OTHER_ISLAND_OUTPUT:[a-z0-9]*]], %{{.*}} = tf_executor.island wraps "tf.D"
      %1:2 = tf_executor.island wraps "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

      %2:2 = tf_executor.island {
        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.A"(%arg0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]], %[[OTHER_ISLAND_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %4 = "tf.B"(%3, %1#0) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[B_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[TPU0_OUTPUT]])
        %5 = "tf.C"(%4) : (tensor<?xi32>) -> tensor<?xi32>
        tf_executor.yield %5 : tensor<?xi32>
      }

      tf_executor.fetch %2#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Single device cluster, no live-in values.

module {
  // CHECK-LABEL: func @nolivein
  func @nolivein() -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"() : () -> tensor<?xi32>
        %3 = "tf.A"() {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: tf_device.return %[[A_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[TPU0_OUTPUT]])
        %4 = "tf.B"(%3) : (tensor<?xi32>) -> tensor<?xi32>
        tf_executor.yield %4 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Multiple clusters of different devices. Clusters depend on each other.

module {
  // CHECK-LABEL: func @multiplerelatedclusters
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @multiplerelatedclusters(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %4 = "tf.C"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[C_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[GPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[TPU0_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.D"(%4) {device = "gpu0"} : (tensor<?xi32>) -> tensor<?xi32>
        // CHECK: tf_device.return %[[D_OUTPUT]]

        // CHECK: tf_executor.yield %[[GPU0_OUTPUT]]
        tf_executor.yield %5 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Multiple clusters of different devices. Clusters do not depend on each other.

module {
  // CHECK-LABEL: func @multipleunrelatedclusters
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @multipleunrelatedclusters(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %4 = "tf.C"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[C_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[GPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.D"(%2) {device = "gpu0"} : (tensor<?xi32>) -> tensor<?xi32>
        // CHECK: tf_device.return %[[D_OUTPUT]]

        // CHECK: %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[TPU0_OUTPUT]], %[[GPU0_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %6 = "tf.E"(%4, %5) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_executor.yield %[[E_OUTPUT]]
        tf_executor.yield %6 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Single device with non-continous instructions in original block.

module {
  // CHECK-LABEL: func @noncontinoussinglecluster
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @noncontinoussinglecluster(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // Note that tf.C is moved before tf_device.launch.
        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[ARG_0]]) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        %4 = "tf.C"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.D"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[D_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[C_OUTPUT]], %[[TPU0_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %6 = "tf.E"(%4, %5) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_executor.yield %[[E_OUTPUT]]
        tf_executor.yield %6 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

// -----

// Multiple device clusters with intertwined instructions in original block.

module {
  // CHECK-LABEL: func @intertwinedclusters
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @intertwinedclusters(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = tf_executor.graph {
      %1:2 = tf_executor.island {

        // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"(%[[ARG_0]])
        %2 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[GPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[ARG_0]]) : (tensor<?xi32>) -> tensor<?xi32>
        // CHECK: tf_device.return %[[C_OUTPUT]]
        // CHECK: {device = "gpu0"} : () -> tensor<?xi32>

        // CHECK: %[[TPU0_OUTPUT:[0-9]*]] = "tf_device.launch"
        // CHECK: %[[B_OUTPUT:[0-9]*]] = "tf.B"(%[[A_OUTPUT]]) : (tensor<?xi32>) -> tensor<?xi32>
        %3 = "tf.B"(%2) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        %4 = "tf.C"(%arg0) {device = "gpu0"} : (tensor<?xi32>) -> tensor<?xi32>

        // CHECK: %[[D_OUTPUT:[0-9]*]] = "tf.D"(%[[A_OUTPUT]], %[[B_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %5 = "tf.D"(%2, %3) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_device.return %[[D_OUTPUT]]
        // CHECK: {device = "tpu0"} : () -> tensor<?xi32>

        // CHECK: %[[E_OUTPUT:[0-9]*]] = "tf.E"(%[[GPU0_OUTPUT]], %[[TPU0_OUTPUT]]) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
        %6 = "tf.E"(%4, %5) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

        // CHECK: tf_executor.yield %[[E_OUTPUT]]
        tf_executor.yield %6 : tensor<?xi32>
      }
      tf_executor.fetch %1#0 : tensor<?xi32>
    }
    return %0 : tensor<?xi32>
  }
}

