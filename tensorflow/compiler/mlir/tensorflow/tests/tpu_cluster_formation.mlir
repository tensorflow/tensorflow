// RUN: tf-opt %s -tf-tpu-cluster-formation | FileCheck %s --dump-input=fail


// Test ops in cluster only have `_tpu_replicate` and `device` attributes
// removed when moved to a launch.
// CHECK-LABEL: func @cluster_ops_removed_attrs
func @cluster_ops_removed_attrs() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate", device = "device", name = "name"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  return
}

// CHECK:      "tf.opA"
// CHECK-SAME: name = "name"
// CHECK-NOT:  _tpu_replicate = "replicate"
// CHECK-NOT:  device = "device"
// CHECK:      "tf_device.return"


// Test TPUReplicateMetadata ops `name` attribute is not copied over to launch.
// CHECK-LABEL: func @launch_no_metadata_op_name_attr
func @launch_no_metadata_op_name_attr() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", name = "name", num_replicas = 1, topology = "topology"} : () -> ()
  return
}

// CHECK-NOT:  name = "name"


// Test TPUReplicateMetadata op is removed when forming clusters.
// CHECK-LABEL: func @metadata_op_removed
func @metadata_op_removed() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  return
}

// CHECK-NOT:  "tf.TPUReplicateMetadata"


// Test ops in an island with the same `_tpu_replicate` attribute are merged
// under a launch.
// CHECK-LABEL: func @simple_island
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @simple_island(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"() : () -> tensor<i1>
      %5 = "tf.opC"(%3) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %5 : tensor<i1>
    }
    tf_executor.fetch %1#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:          "tf.opB"
// CHECK:          %[[LAUNCH:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:       %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:       "tf_device.return"(%[[OP_C]])
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
// CHECK-SAME:     num_replicas = 1
// CHECK-SAME:     topology = "topology"
// CHECK:          tf_executor.yield %[[LAUNCH]]


// Test ops in an island with the same `_tpu_replicate` attribute are merged
// under a launch, even when the associated TPUReplicateMetadata op is in a
// different island.
// CHECK-LABEL: func @simple_island_separate_metadata
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @simple_island_separate_metadata(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1 = tf_executor.island {
      "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
    }
    %2:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"() : () -> tensor<i1>
      %5 = "tf.opC"(%3) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %5 : tensor<i1>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK:          "tf.opB"
// CHECK:          %[[LAUNCH:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:       %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:       "tf_device.return"(%[[OP_C]])
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
// CHECK-SAME:     num_replicas = 1
// CHECK-SAME:     topology = "topology"
// CHECK:          tf_executor.yield %[[LAUNCH]]


// Test ops in multiple islands with the same `_tpu_replicate` attribute are
// merged under launch ops only within their respective island.
// CHECK-LABEL: func @multiple_islands_separate_metadata
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @multiple_islands_separate_metadata(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  %0:2 = tf_executor.graph {
    %1 = tf_executor.island {
      "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
    }
    %2:2 = tf_executor.island {
      %3 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opB"() : () -> tensor<i1>
      %5 = "tf.opC"(%3) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %5 : tensor<i1>
    }
    %6:2 = tf_executor.island {
      %7 = "tf.opD"(%2#0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      %8 = "tf.opE"() : () -> tensor<i1>
      %9 = "tf.opF"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %9 : tensor<i1>
    }
    tf_executor.fetch %2#0, %6#0 : tensor<i1>, tensor<i1>
  }
  return %0#0, %0#1 : tensor<i1>, tensor<i1>
}

// CHECK:        %[[ISLAND_1:[0-9]*]]:2 = tf_executor.island {
// CHECK:          "tf.opB"
// CHECK:          %[[LAUNCH_0:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:       %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:       "tf_device.return"(%[[OP_C]])
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
// CHECK-SAME:     num_replicas = 1
// CHECK-SAME:     topology = "topology"
// CHECK:          tf_executor.yield %[[LAUNCH_0]]
// CHECK:        tf_executor.island {
// CHECK:          "tf.opE"
// CHECK:          %[[LAUNCH_1:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_D:[0-9]*]] = "tf.opD"(%[[ISLAND_1]]#0)
// CHECK-NEXT:       %[[OP_F:[0-9]*]] = "tf.opF"(%[[ARG_0]])
// CHECK-NEXT:       "tf_device.return"(%[[OP_F]])
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
// CHECK-SAME:     num_replicas = 1
// CHECK-SAME:     topology = "topology"
// CHECK:          tf_executor.yield %[[LAUNCH_1]]


// Test ops in a function body with the same `_tpu_replicate` attribute are
// merged under a launch op.
// CHECK-LABEL: func @ops_in_func_body
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @ops_in_func_body(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() : () -> tensor<i1>
  %2 = "tf.opC"(%0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%2) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"() : () -> tensor<i1>
  %5 = "tf.opF"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  return %2, %3, %5 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK:      "tf.opB"
// CHECK:      "tf.opE"
// CHECK:      %[[LAUNCH:[0-9]*]]:3 = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:   %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_C]])
// CHECK-NEXT:   %[[OP_F:[0-9]*]] = "tf.opF"(%[[ARG_0]])
// CHECK-NEXT:   "tf_device.return"(%[[OP_C]], %[[OP_D]], %[[OP_F]])
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: num_replicas = 1
// CHECK-SAME: topology = "topology"
// CHECK:      return %[[LAUNCH]]#0, %[[LAUNCH]]#1, %[[LAUNCH]]#2


// Test a nested user of an op in a cluster has its operand be updated to launch
// result.
// CHECK-LABEL: func @nested_cluster_op_user
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @nested_cluster_op_user(%arg0 : tensor<i1>) -> (tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %1 = tf_executor.graph {
    tf_executor.fetch %0 : tensor<i1>
  }
  %2 = "tf.opB"(%0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  return %2 : tensor<i1>
}

// CHECK:      %[[LAUNCH:[0-9]*]]:2 = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT:   "tf_device.return"(%[[OP_A]], %[[OP_B]])
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: num_replicas = 1
// CHECK-SAME: topology = "topology"
// CHECK:      tf_executor.graph {
// CHECK-NEXT:   tf_executor.fetch %[[LAUNCH]]#0
// CHECK:      return %[[LAUNCH]]#1


// Test nested op of a cluster with an operand from an op of the same cluster
// retains its original operand.
// CHECK-LABEL: func @nested_cluster_op
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @nested_cluster_op(%arg0 : tensor<i1>) -> (tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() ( {
    "tf.opC"(%0) : (tensor<i1>) -> tensor<i1>
  }) {_tpu_replicate = "replicate"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  return %1 : tensor<i1>
}

// CHECK:      %[[LAUNCH:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"() ( {
// CHECK-NEXT:     "tf.opC"(%[[OP_A]])
// CHECK:        "tf_device.return"(%[[OP_B]])
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: num_replicas = 1
// CHECK-SAME: topology = "topology"
// CHECK:      return %[[LAUNCH]]


// Test multiple clusters interleaved.
// CHECK-LABEL: func @interleaved_clusters
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @interleaved_clusters(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate_1", device = "device_1", num_replicas = 1, topology = "topology_1"} : () -> ()
  %0 = "tf.opA"(%arg0) {_tpu_replicate = "replicate_0"} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"(%arg0) {_tpu_replicate = "replicate_1"} : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"(%0) {_tpu_replicate = "replicate_0"} : (tensor<i1>) -> tensor<i1>
  %3 = "tf.opD"(%1) {_tpu_replicate = "replicate_1"} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate_0", device = "device_0", num_replicas = 0, topology = "topology_0"} : () -> ()
  return %2, %3 : tensor<i1>, tensor<i1>
}

// CHECK:      %[[LAUNCH_0:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:   "tf_device.return"(%[[OP_C]])
// CHECK-NEXT: _tpu_replicate = "replicate_0"
// CHECK-SAME: device = "device_0"
// CHECK-SAME: num_replicas = 0
// CHECK-SAME: topology = "topology_0"
// CHECK:      %[[LAUNCH_1:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_B]])
// CHECK-NEXT:   "tf_device.return"(%[[OP_D]])
// CHECK-NEXT: _tpu_replicate = "replicate_1"
// CHECK-SAME: device = "device_1"
// CHECK-SAME: num_replicas = 1
// CHECK-SAME: topology = "topology_1"
// CHECK:      return %[[LAUNCH_0]], %[[LAUNCH_1]]


// Test operands and results of ops of a cluster that are interleaved between
// other ops of the same cluster are moved before and after the cluster
// properly.
// CHECK-LABEL: func @interleaved_cluster_operands_results
func @interleaved_cluster_operands_results() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate"} : () -> tensor<i1>
  %1 = "tf.opB"(%0) : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"() : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%1) : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"(%2) : (tensor<i1>) -> tensor<i1>
  %5 = "tf.opF"(%4) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  return
}

// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_C]])
// CHECK:      %[[LAUNCH:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"
// CHECK-NEXT:   "tf.opF"(%[[OP_E]])
// CHECK-NEXT:   "tf_device.return"(%[[OP_A]])
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: num_replicas = 1
// CHECK-SAME: topology = "topology"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"(%[[LAUNCH]])
// CHECK:      "tf.opD"(%[[OP_B]])
