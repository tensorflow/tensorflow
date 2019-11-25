// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-cluster-formation | FileCheck %s --dump-input=fail


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
// CHECK:      tf_device.return


// Test TPUReplicateMetadata ops `name` and `num_replicas` attributes are not
// copied over to launch.
// CHECK-LABEL: func @launch_removed_metadata_attrs
func @launch_removed_metadata_attrs() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", name = "name", num_replicas = 1, topology = "topology"} : () -> ()
  return
}

// CHECK-NOT:  name = "name"
// CHECK-NOT:  num_replicas = 1


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
// CHECK-NEXT:       tf_device.return %[[OP_C]]
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
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
// CHECK-NEXT:       tf_device.return %[[OP_C]]
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
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

// CHECK:        %[[ISLAND_1:.*]], %[[ISLAND_1_control:.*]] = tf_executor.island {
// CHECK:          "tf.opB"
// CHECK:          %[[LAUNCH_0:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:       %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:       tf_device.return %[[OP_C]]
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
// CHECK-SAME:     topology = "topology"
// CHECK:          tf_executor.yield %[[LAUNCH_0]]
// CHECK:        tf_executor.island {
// CHECK:          "tf.opE"
// CHECK:          %[[LAUNCH_1:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:       %[[OP_D:[0-9]*]] = "tf.opD"(%[[ISLAND_1]])
// CHECK-NEXT:       %[[OP_F:[0-9]*]] = "tf.opF"(%[[ARG_0]])
// CHECK-NEXT:       tf_device.return %[[OP_F]]
// CHECK-NEXT:     _tpu_replicate = "replicate"
// CHECK-SAME:     device = "device"
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
// CHECK-NEXT:   tf_device.return %[[OP_C]], %[[OP_D]], %[[OP_F]]
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
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
// CHECK-NEXT:   tf_device.return %[[OP_A]], %[[OP_B]]
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
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
// CHECK:        tf_device.return %[[OP_B]]
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
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
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate_0", device = "device_0", num_replicas = 1, topology = "topology_0"} : () -> ()
  return %2, %3 : tensor<i1>, tensor<i1>
}

// CHECK:      %[[LAUNCH_0:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:   tf_device.return %[[OP_C]]
// CHECK-NEXT: _tpu_replicate = "replicate_0"
// CHECK-SAME: device = "device_0"
// CHECK-SAME: topology = "topology_0"
// CHECK:      %[[LAUNCH_1:[0-9]*]] = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_B]])
// CHECK-NEXT:   tf_device.return %[[OP_D]]
// CHECK-NEXT: _tpu_replicate = "replicate_1"
// CHECK-SAME: device = "device_1"
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
// CHECK-NEXT:   tf_device.return %[[OP_A]]
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: topology = "topology"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"(%[[LAUNCH]])
// CHECK:      "tf.opD"(%[[OP_B]])


// Test one replica cluster results in removing of TPUReplicatedInput and
// TPUReplicatedOutput nodes and operands are forwarded to results.
// CHECK-LABEL: func @one_replica
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func @one_replica(%arg0: tensor<i1>) -> tensor<i1> {
  %ri = "tf.TPUReplicatedInput"(%arg0) : (tensor<i1>) -> tensor<i1>
  %0 = "tf.opA"(%ri) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"(%0) : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"() : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%1) : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"(%2) : (tensor<i1>) -> tensor<i1>
  %5 = "tf.opF"(%4) {_tpu_replicate = "replicate"} : (tensor<i1>) -> tensor<i1>
  %ro = "tf.TPUReplicatedOutput"(%5) : (tensor<i1>) -> tensor<i1>
  return %ro : tensor<i1>
}

// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_C]])
// CHECK:      %[[LAUNCH:[0-9]*]]:2 = "tf_device.launch"() ( {
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_F:[0-9]*]] = "tf.opF"(%[[OP_E]])
// CHECK-NEXT:   tf_device.return %[[OP_A]], %[[OP_F]]
// CHECK-NEXT: _tpu_replicate = "replicate"
// CHECK-SAME: device = "device"
// CHECK-SAME: topology = "topology"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"(%[[LAUNCH]]#0)
// CHECK:      "tf.opD"(%[[OP_B]])
// CHECK:      return %[[LAUNCH]]#1
// CHECK-NOT:  "tf.TPUReplicatedInput"
// CHECK-NOT:  "tf.TPUReplicatedOutput"


// Test replication with replicated operands and replicated results. The cluster
// will be wrapped in a launch first and then by a replicate. TPUReplicatedInput
// and TPUReplicatedOutput nodes will be replaced by the replicate operands and
// results.
// CHECK-LABEL: func @replication
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>, %[[ARG_2:[a-z0-9]*]]: tensor<f32>)
func @replication(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %0 = "tf.opA"() : () -> tensor<i1>
  %ri_0 = "tf.TPUReplicatedInput"(%arg0, %0) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() : () -> tensor<i32>
  %ri_1 = "tf.TPUReplicatedInput"(%1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.opC"() : () -> tensor<f32>
  %3 = "tf.opD"(%ri_0, %ri_1, %arg2, %2) {_tpu_replicate = "replicate"} : (tensor<i1>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<i32>
  %ro_0:2 = "tf.TPUReplicatedOutput"(%3) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 2, topology = "topology"} : () -> ()
  %7 = "tf.opE"(%3, %ri_0, %ri_1, %arg2, %2) {_tpu_replicate = "replicate"} : (tensor<i32>, tensor<i1>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %ro_1:2 = "tf.TPUReplicatedOutput"(%7) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %ro_0#0, %ro_1#1 : tensor<i32>, tensor<f32>
}

// CHECK:      %[[OP_A:[0-9]*]] = "tf.opA"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"
// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
// CHECK-SAME: ([%[[ARG_0]], %[[OP_A]]] as %[[RI_0:[a-z0-9]*]]: tensor<i1>, [%[[OP_B]], %[[ARG_1]]] as %[[RI_1:[a-z0-9]*]]: tensor<i32>)
// CHECK-SAME: n = 2 : i32
// CHECK-NEXT:   %[[LAUNCH:[0-9]*]]:2 = "tf_device.launch"() ( {
// CHECK:          %[[OP_D:[0-9]*]] = "tf.opD"(%[[RI_0]], %[[RI_1]], %[[ARG_2]], %[[OP_C]])
// CHECK:          %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]], %[[RI_0]], %[[RI_1]], %[[ARG_2]], %[[OP_C]])
// CHECK:          tf_device.return %[[OP_D]], %[[OP_E]]
// CHECK-NEXT:   _tpu_replicate = "replicate"
// CHECK-SAME:   device = "device"
// CHECK-SAME:   topology = "topology"
// CHECK:        tf_device.return %[[LAUNCH]]#0, %[[LAUNCH]]#1
// CHECK:      return %[[REPLICATE]]#0, %[[REPLICATE]]#3


// -----


// Test cluster with missing `num_replicas` attribute.
func @missing_num_replicas() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate", device = "device", name = "name"} : () -> tensor<i1>
  // expected-error@+1 {{'tf.TPUReplicateMetadata' op requires attribute 'num_replicas'}}
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", topology = "topology"} : () -> ()
  return
}


// -----


// Test cluster with bad `num_replicas` attribute.
func @bad_num_replicas() {
  // expected-error@+1 {{requires 'num_replicas' int attribute to be at least 1}}
  %0 = "tf.opA"() {_tpu_replicate = "replicate", device = "device", name = "name"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 0, topology = "topology"} : () -> ()
  return
}


// -----


// Test cluster with TPUReplicatedInput where the number of operands does not
// match associated `num_replicas` attribute.
func @mismatched_replicated_input(%arg0: tensor<i1>) {
  // expected-error@+1 {{'tf.TPUReplicatedInput' op requires 2 operands}}
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0, %arg0) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
  %1 = "tf.opA"(%0) {_tpu_replicate = "replicate", device = "device", name = "name"} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 2, topology = "topology"} : () -> ()
  return
}


// -----


// Test cluster with TPUReplicatedOutput where the number of results does not
// match associated `num_replicas` attribute.
func @mismatched_replicated_output() {
  %0 = "tf.opA"() {_tpu_replicate = "replicate", device = "device", name = "name"} : () -> tensor<i1>
  // expected-error@+1 {{'tf.TPUReplicatedOutput' op requires 2 results}}
  %1:3 = "tf.TPUReplicatedOutput"(%0) : (tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>)
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 2, topology = "topology"} : () -> ()
  return
}


// -----


// Test cluster that should be replicated where its outputs do not lead to a
// TPUReplicatedOutput.
func @missing_replicated_output() {
  // expected-error@+1 {{requires output of tf_device.launch to lead to a 'tf.TPUReplicatedOutput' op}}
  %0 = "tf.opA"() {_tpu_replicate = "replicate", device = "device", name = "name"} : () -> tensor<i1>
  %1 = "tf.opB"(%0) : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 2, topology = "topology"} : () -> ()
  return
}


// -----


// Test unused TPUReplicatedInput that has more than one operand.
func @leftover_replicated_input(%arg0: tensor<i1>) {
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return
}


// -----


// Test unused TPUReplicatedOutput that has more than one result.
func @leftover_replicated_output(%arg0: tensor<i1>) {
  %0:2 = "tf.TPUReplicatedOutput"(%arg0) : (tensor<i1>) -> (tensor<i1>, tensor<i1>)
  return
}
