// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-cluster-formation | FileCheck %s


// Test ops in cluster only have `_replication_info` and `device` attributes
// removed when moved to a `tf_device.cluster`.
// CHECK-LABEL: func @cluster_ops_removed_attrs
func.func @cluster_ops_removed_attrs() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK:      "tf.opA"
// CHECK-SAME: name = "name"
// CHECK-NOT:  _replication_info = "replicate"
// CHECK-NOT:  device = "/device:TPU:0"
// CHECK:      tf_device.return


// Test TPUReplicateMetadata ops `name` and `num_replicas` attributes are not
// copied over to `tf_device.cluster`.
// CHECK-LABEL: func @removed_metadata_attrs
func.func @removed_metadata_attrs() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK-NOT:  name = "name"
// CHECK-NOT:  num_replicas = 1


// Test TPUReplicateMetadata op is removed when forming clusters.
// CHECK-LABEL: func @metadata_op_removed
func.func @metadata_op_removed() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK-NOT:  "tf.TPUReplicateMetadata"


// Test ops in a function body with the same `_replication_info` attribute are
// merged under a `tf_device.cluster` op.
// CHECK-LABEL: func @ops_in_func_body
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func.func @ops_in_func_body(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() {is_stateless = true} : () -> tensor<i1>
  %2 = "tf.opC"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%2) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"() {is_stateless = true} : () -> tensor<i1>
  %5 = "tf.opF"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  func.return %2, %3, %5 : tensor<i1>, tensor<i1>, tensor<i1>
}

// CHECK:      "tf.opB"
// CHECK:      "tf.opE"
// CHECK:      %[[CLUSTER:[0-9]*]]:3 = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:   %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_C]])
// CHECK-NEXT:   %[[OP_F:[0-9]*]] = "tf.opF"(%[[ARG_0]])
// CHECK-NEXT:   tf_device.return %[[OP_C]], %[[OP_D]], %[[OP_F]]
// CHECK-NEXT: _replication_info = "replicate"
// CHECK-SAME: device = "/device:TPU:0"
// CHECK-SAME: topology = "topology"
// CHECK:      return %[[CLUSTER]]#0, %[[CLUSTER]]#1, %[[CLUSTER]]#2


// Test a nested user of an op in a cluster has its operand be updated to
// `tf_device.cluster` result.
// CHECK-LABEL: func @nested_cluster_op_user
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func.func @nested_cluster_op_user(%arg0 : tensor<i1>) -> (tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %1 = "tf_device.launch"() ({
    tf_device.return %0 : tensor<i1>
  }) {device = "/device:TPU:0"} : () -> tensor<i1>
  %2 = "tf.opB"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return %2 : tensor<i1>
}

// CHECK:      %[[CLUSTER:[0-9]*]]:2 = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[OP_A]])
// CHECK-NEXT:   tf_device.return %[[OP_A]], %[[OP_B]]
// CHECK-NEXT: _replication_info = "replicate"
// CHECK-SAME: device = "/device:TPU:0"
// CHECK-SAME: topology = "topology"
// CHECK:      tf_device.launch
// CHECK-NEXT:   tf_device.return %[[CLUSTER]]#0
// CHECK:      return %[[CLUSTER]]#1


// Test nested op of a cluster with an operand from an op of the same cluster
// retains its original operand.
// CHECK-LABEL: func @nested_cluster_op
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func.func @nested_cluster_op(%arg0 : tensor<i1>) -> (tensor<i1>) {
  %0 = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() ({
    "tf.opC"(%0) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  }) {_xla_compile_device_type = "TPU", _replication_info = "replicate"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return %1 : tensor<i1>
}

// CHECK:      %[[CLUSTER:[0-9]*]] = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"() ({
// CHECK-NEXT:     "tf.opC"(%[[OP_A]])
// CHECK:        tf_device.return %[[OP_B]]
// CHECK-NEXT: _replication_info = "replicate"
// CHECK-SAME: device = "/device:TPU:0"
// CHECK-SAME: topology = "topology"
// CHECK:      return %[[CLUSTER]]


// Test multiple clusters interleaved.
// CHECK-LABEL: func @interleaved_clusters
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func.func @interleaved_clusters(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", device = "device_1", num_replicas = 1, topology = "topology_1"} : () -> ()
  %0 = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %3 = "tf.opD"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", device = "device_0", num_replicas = 1, topology = "topology_0"} : () -> ()
  func.return %2, %3 : tensor<i1>, tensor<i1>
}

// CHECK:      %[[CLUSTER_0:[0-9]*]] = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_C:[0-9]*]] = "tf.opC"(%[[OP_A]])
// CHECK-NEXT:   tf_device.return %[[OP_C]]
// CHECK-NEXT: _replication_info = "replicate_0"
// CHECK-SAME: device = "device_0"
// CHECK-SAME: topology = "topology_0"
// CHECK:      %[[CLUSTER_1:[0-9]*]] = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_B:[0-9]*]] = "tf.opB"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_D:[0-9]*]] = "tf.opD"(%[[OP_B]])
// CHECK-NEXT:   tf_device.return %[[OP_D]]
// CHECK-NEXT: _replication_info = "replicate_1"
// CHECK-SAME: device = "device_1"
// CHECK-SAME: topology = "topology_1"
// CHECK:      return %[[CLUSTER_0]], %[[CLUSTER_1]]


// Test operands and results of ops of a cluster that are interleaved between
// other ops of the same cluster are moved before and after the cluster
// properly.
// CHECK-LABEL: func @interleaved_cluster_operands_results
func.func @interleaved_cluster_operands_results() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : () -> tensor<i1>
  %1 = "tf.opB"(%0) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"() {is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%1) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"(%2) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.opF"(%4) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  func.return
}

// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_C]])
// CHECK:      %[[CLUSTER:[0-9]*]] = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"
// CHECK-NEXT:   "tf.opF"(%[[OP_E]])
// CHECK-NEXT:   tf_device.return %[[OP_A]]
// CHECK-NEXT: _replication_info = "replicate"
// CHECK-SAME: device = "/device:TPU:0"
// CHECK-SAME: topology = "topology"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"(%[[CLUSTER]])
// CHECK:      "tf.opD"(%[[OP_B]])


// Test one replica cluster results in removing of TPUReplicatedInput and
// TPUReplicatedOutput nodes and operands are forwarded to results.
// CHECK-LABEL: func @one_replica
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>)
func.func @one_replica(%arg0: tensor<i1>) -> tensor<i1> {
  %ri = "tf.TPUReplicatedInput"(%arg0) : (tensor<i1>) -> tensor<i1>
  %0 = "tf.opA"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"(%0) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %2 = "tf.opC"() {is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  %3 = "tf.opD"(%1) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.opE"(%2) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.opF"(%4) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  %ro = "tf.TPUReplicatedOutput"(%5) : (tensor<i1>) -> tensor<i1>
  func.return %ro : tensor<i1>
}

// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_C]])
// CHECK:      %[[CLUSTER:[0-9]*]]:2 = "tf_device.cluster"() ({
// CHECK-NEXT:   %[[OP_A:[0-9]*]] = "tf.opA"(%[[ARG_0]])
// CHECK-NEXT:   %[[OP_F:[0-9]*]] = "tf.opF"(%[[OP_E]])
// CHECK-NEXT:   tf_device.return %[[OP_A]], %[[OP_F]]
// CHECK-NEXT: _replication_info = "replicate"
// CHECK-SAME: device = "/device:TPU:0"
// CHECK-SAME: topology = "topology"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"(%[[CLUSTER]]#0)
// CHECK:      "tf.opD"(%[[OP_B]])
// CHECK:      return %[[CLUSTER]]#1
// CHECK-NOT:  "tf.TPUReplicatedInput"
// CHECK-NOT:  "tf.TPUReplicatedOutput"


// Test replication with replicated operands and replicated results. The cluster
// will be wrapped in a `tf_device.cluster` first and then by a replicate.
// TPUReplicatedInput and TPUReplicatedOutput nodes will be replaced by the
// replicate operands and results.
// CHECK-LABEL: func @replication
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>, %[[ARG_2:[a-z0-9]*]]: tensor<f32>)
func.func @replication(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %0 = "tf.opA"() {is_stateless = true} : () -> tensor<i1>
  %ri_0 = "tf.TPUReplicatedInput"(%arg0, %0) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %1 = "tf.opB"() {is_stateless = true} : () -> tensor<i32>
  %ri_1 = "tf.TPUReplicatedInput"(%1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.opC"() {is_stateless = true} : () -> tensor<f32>
  %3 = "tf.opD"(%ri_0, %ri_1, %arg2, %2) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<i32>
  %ro_0:2 = "tf.TPUReplicatedOutput"(%3) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  %7 = "tf.opE"(%3, %ri_0, %ri_1, %arg2, %2) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i32>, tensor<i1>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %ro_1:2 = "tf.TPUReplicatedOutput"(%7) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %ro_0#0, %ro_1#1 : tensor<i32>, tensor<f32>
}

// CHECK:      %[[OP_A:[0-9]*]] = "tf.opA"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"
// CHECK:      %[[OP_C:[0-9]*]] = "tf.opC"
// CHECK:      %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
// CHECK-DAG:  [%[[ARG_0]], %[[OP_A]]] as %[[RI_0:[a-z0-9]*]]: tensor<i1>
// CHECK-DAG:  [%[[OP_B]], %[[ARG_1]]] as %[[RI_1:[a-z0-9]*]]: tensor<i32>
// CHECK-SAME: n = 2 : i32
// CHECK-NEXT:   %[[CLUSTER:[0-9]*]]:2 = "tf_device.cluster"() ({
// CHECK:          %[[OP_D:[0-9]*]] = "tf.opD"(%[[RI_0]], %[[RI_1]], %[[ARG_2]], %[[OP_C]])
// CHECK:          %[[OP_E:[0-9]*]] = "tf.opE"(%[[OP_D]], %[[RI_0]], %[[RI_1]], %[[ARG_2]], %[[OP_C]])
// CHECK:          tf_device.return %[[OP_D]], %[[OP_E]]
// CHECK-NEXT:   _replication_info = "replicate"
// CHECK-SAME:   device = "/device:TPU:0"
// CHECK-SAME:   topology = "topology"
// CHECK:        tf_device.return %[[CLUSTER]]#0, %[[CLUSTER]]#1
// CHECK:      return %[[REPLICATE]]#0, %[[REPLICATE]]#3


// Test replication with model parallelism using partitioned resource inputs.
// The cluster will be wrapped in a `tf_device.cluster` first and then by a
// replicate.
// TPUPartitionedInputV2 nodes would be inside the replicate but outside the
// cluster.
// TPUReplicatedInput and TPUReplicatedOutput nodes will be replaced by the
// replicate operands and results.
// CHECK-LABEL: func @replication_with_model_parallelism
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, %[[ARG_1:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, %[[ARG_2:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>, %[[ARG_3:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>)
!rtype = tensor<!tf_type.resource<tensor<10x3xf32>>>

func.func @replication_with_model_parallelism(%arg0: !rtype, %arg1: !rtype, %arg2: !rtype, %arg3: !rtype) -> (tensor<10x3xf32>, tensor<f32>) {
  %0 = "tf.opA"() {is_stateless = true} : () -> tensor<i32>
  %1 = "tf.opB"() {is_stateless = true} : () -> tensor<i32>
  %2 = "tf.TPUReplicatedInput"(%arg0, %arg2) : (!rtype, !rtype) -> !rtype
  %3 = "tf.TPUReplicatedInput"(%arg1, %arg3) : (!rtype, !rtype) -> !rtype
  %4 = "tf.TPUPartitionedInputV2"(%2, %3) {_XlaSharding = "", device = "", partition_dims = []} : (!rtype, !rtype) -> !rtype
  %5 = "tf.TPUReplicatedInput"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %6 = "tf.opC"(%4) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (!rtype) -> tensor<10x3xf32>
  %7:2 = "tf.TPUReplicatedOutput"(%6) : (tensor<10x3xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_cores_per_replica = 2 : i64, num_replicas = 2 : i64, topology = "topology"} : () -> ()
  %8 = "tf.opD"(%5) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i32>) -> tensor<f32>
  %9:2 = "tf.TPUReplicatedOutput"(%8) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %7#0, %9#1 : tensor<10x3xf32>, tensor<f32>
}

// CHECK:      %[[OP_A:[0-9]*]] = "tf.opA"
// CHECK:      %[[OP_B:[0-9]*]] = "tf.opB"
// CHECK:      %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
// CHECK-DAG:  [%[[ARG_0]], %[[ARG_2]]] as %[[RI_0:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>
// CHECK-DAG:  [%[[ARG_1]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<!tf_type.resource<tensor<10x3xf32>>>
// CHECK-DAG:  [%[[OP_A]], %[[OP_B]]] as %[[RI_2:[a-z0-9]*]]: tensor<i32>
// CHECK-SAME: n = 2 : i32
// CHECK:        %[[PI:[0-9]*]] = "tf.TPUPartitionedInputV2"(%[[RI_0]], %[[RI_1]])
// CHECK-NEXT:   %[[CLUSTER:[0-9]*]]:2 = "tf_device.cluster"() ({
// CHECK:          %[[OP_C:[0-9]*]] = "tf.opC"(%[[PI]])
// CHECK:          %[[OP_D:[0-9]*]] = "tf.opD"(%[[RI_2]])
// CHECK:          tf_device.return %[[OP_C]], %[[OP_D]]
// CHECK-NEXT:   _replication_info = "replicate"
// CHECK-SAME:   device = "/device:TPU:0"
// CHECK-SAME:   topology = "topology"
// CHECK:        tf_device.return %[[CLUSTER]]#0, %[[CLUSTER]]#1
// CHECK:      return %[[REPLICATE]]#0, %[[REPLICATE]]#3


// Test TPUReplicatedInputs with non contiguous `index` attributes.
// CHECK-LABEL: func @non_contigous_indices
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<i1>, %[[ARG_1:.*]]: tensor<i1>, %[[ARG_2:.*]]: tensor<i1>, %[[ARG_3:.*]]: tensor<i1>, %[[ARG_4:.*]]: tensor<i1>, %[[ARG_5:.*]]: tensor<i1>)
func.func @non_contigous_indices(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<i1>, %arg5: tensor<i1>) {
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0) {index = 8 : i64} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  "tf.opA"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  %1 = "tf.TPUReplicatedInput"(%arg1) {index = 6 : i64, is_packed = true} : (tensor<i1>) -> tensor<i1>
  "tf.opA"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  %2 = "tf.TPUReplicatedInput"(%arg2, %arg2) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  "tf.opB"(%2) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  %3 = "tf.TPUReplicatedInput"(%arg3) {is_packed = true} : (tensor<i1>) -> tensor<i1>
  "tf.opB"(%3) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  %4 = "tf.TPUReplicatedInput"(%arg4, %arg4) {index = 2 : i64} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  "tf.opC"(%4) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  %5 = "tf.TPUReplicatedInput"(%arg5) {index = 4 : i64, is_packed = true} : (tensor<i1>) -> tensor<i1>
  "tf.opC"(%5) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}

// CHECK:      tf_device.replicate
// CHECK-SAME: [%[[ARG_0]], %[[ARG_0]]] as %{{[a-z0-9]*}}
// CHECK-SAME: [%[[ARG_2]], %[[ARG_2]]] as %{{[a-z0-9]*}}
// CHECK-SAME: [%[[ARG_4]], %[[ARG_4]]] as %{{[a-z0-9]*}}
// CHECK-SAME: %[[ARG_1]] as %{{[a-z0-9]*}}
// CHECK-SAME: %[[ARG_3]] as %{{[a-z0-9]*}}
// CHECK-SAME: %[[ARG_5]] as %{{[a-z0-9]*}}


// Test that the `is_mirrored_variable` attribute is preserved in the
// tf_device.replicate op.
// CHECK-LABEL: func @mirrored_variables
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<!tf_type.resource<tensor<32xf32>>>, %[[ARG_1:.*]]: tensor<!tf_type.resource<tensor<32xf32>>>, %[[ARG_2:.*]]: tensor<!tf_type.resource<tensor<32xf32>>>, %[[ARG_3:.*]]: tensor<!tf_type.resource<tensor<32xf32>>>, %[[ARG_4:.*]]: tensor<!tf_type.resource<tensor<32xf32>>>)
func.func @mirrored_variables(%arg0: tensor<!tf_type.resource<tensor<32xf32>>>, %arg1: tensor<!tf_type.resource<tensor<32xf32>>>, %arg2: tensor<!tf_type.resource<tensor<32xf32>>>, %arg3: tensor<!tf_type.resource<tensor<32xf32>>>, %arg4: tensor<!tf_type.resource<tensor<32xf32>>>) {
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg1) {index = 0 : i64} : (tensor<!tf_type.resource<tensor<32xf32>>>, tensor<!tf_type.resource<tensor<32xf32>>>) -> tensor<!tf_type.resource<tensor<32xf32>>>
  %1 = "tf.TPUReplicatedInput"(%arg2, %arg3) {index = 1 : i64, is_mirrored_variable = true} : (tensor<!tf_type.resource<tensor<32xf32>>>, tensor<!tf_type.resource<tensor<32xf32>>>) -> tensor<!tf_type.resource<tensor<32xf32>>>
  %2 = "tf.TPUReplicatedInput"(%arg4) {index = 2 : i64, is_mirrored_variable = true, is_packed = true} : (tensor<!tf_type.resource<tensor<32xf32>>>) -> tensor<!tf_type.resource<tensor<32xf32>>>
  "tf.opA"(%0, %1, %2) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", is_stateless = true} : (tensor<!tf_type.resource<tensor<32xf32>>>, tensor<!tf_type.resource<tensor<32xf32>>>, tensor<!tf_type.resource<tensor<32xf32>>>) -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}

// CHECK:      tf_device.replicate
// CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %{{[a-z0-9]*}}
// CHECK-SAME: %[[ARG_4]] as %{{[a-z0-9]*}}
// CHECK-SAME: _mirrored_variable_indices = [1, 2]


// Test resource usage after resource use in cluster is moved to after the
// cluster.
// CHECK-LABEL: func @resource_after_cluster
// CHECK-SAME:  ([[USED_RESOURCE:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>, [[UNUSED_RESOURCE:%.*]]: tensor<*x!tf_type.resource<tensor<f32>>>)
func.func @resource_after_cluster(%arg0: tensor<*x!tf_type.resource<tensor<f32>>>, %arg1: tensor<*x!tf_type.resource<tensor<f32>>>) {
  // CHECK-NEXT: [[CONST:%.*]] = "tf.Const"
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>

  // CHECK-NEXT: "tf.AssignSubVariableOp"([[UNUSED_RESOURCE]], [[CONST]])

  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.ReadVariableOp"([[USED_RESOURCE]])
  // CHECK-NEXT:   "tf.NoOp"
  // CHECK-NEXT:   tf_device.return
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster_test_fn", allow_soft_placement = false, computation_shape = [], device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
  %1 = "tf.ReadVariableOp"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster_test_fn"} : (tensor<*x!tf_type.resource<tensor<f32>>>) -> tensor<f32>

  "tf.AssignSubVariableOp"(%arg1, %0) : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()

  // CHECK:       "tf.AssignAddVariableOp"([[USED_RESOURCE]], [[CONST]])
  "tf.AssignAddVariableOp"(%arg0, %0) : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()

  "tf.NoOp"() {_xla_compile_device_type = "TPU", _replication_info = "cluster_test_fn"} : () -> ()
  func.return
}


// Test resource not used by cluster is moved to before the cluster.
// CHECK-LABEL: func @resource_before_cluster
func.func @resource_before_cluster() {
  // CHECK-NEXT: [[CONST:%.*]] = "tf.Const"
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>

  // CHECK-NEXT: [[UNUSED_RESOURCE:%.*]] = "tf.VarHandleOp"
  // CHECK-NEXT: "tf.AssignAddVariableOp"([[UNUSED_RESOURCE]], [[CONST]])

  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.NoOp"
  // CHECK-NEXT:   tf_device.return
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster_test_fn", allow_soft_placement = false, computation_shape = [], device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 1 : i64, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()

  %1 = "tf.VarHandleOp"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<*x!tf_type.resource<tensor<f32>>>
  "tf.AssignAddVariableOp"(%1, %0) : (tensor<*x!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()

  "tf.NoOp"() {_xla_compile_device_type = "TPU", _replication_info = "cluster_test_fn"} : () -> ()
  func.return
}


// Test cluster formation with ops with attached regions within a cluster.
// Nested op's that are moved should get their _replication_info and device
// attributes cleared.
// CHECK-LABEL: func @cluster_ops_with_regions
func.func @cluster_ops_with_regions() {
  %0 = "tf.opA"() ({
      %1 = "tf.opB"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "nameB", is_stateless = true} : () -> (tensor<i32>)
    }) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "nameA"} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK:      "tf.opA"() ({
// CHECK-NEXT: "tf.opB"
// CHECK-NOT: _replication_info = "replicate"
// CHECK-NOT:  device = "/device:TPU:0"
// CHECK-SAME: name = "nameB"
// CHECK:      })
// CHECK-NOT: _replication_info = "replicate"
// CHECK-NOT:  device = "/device:TPU:0"
// CHECK:      name = "nameA"
// CHECK:      tf_device.return

// A nested cluster op using result of another cluster op. In the below, opA and
// opB go in a cluster, and opD stays outside.
// CHECK-LABEL: func @cluster_nested_op_using_other_op
func.func @cluster_nested_op_using_other_op() {
  %0 = "tf.opA"() { _xla_compile_device_type = "TPU", _replication_info = "foo" , is_stateless = true} : () -> tensor<i32>
  "tf.opB"() ({
    "tf.opC"(%0) {is_stateless = true} : (tensor<i32>) -> ()
   }) { _xla_compile_device_type = "TPU", _replication_info = "foo" } : () -> ()
  "tf.opD"(%0) {is_stateless = true} : (tensor<i32>) -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "foo", device = "CPU", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK: [[CLUSTER:%.*]] = "tf_device.cluster"() ({
// CHECK:    [[OPA:%.*]] = "tf.opA"() {is_stateless = true} : () -> tensor<i32>
// CHECK:    "tf.opB"() ({
// CHECK:      "tf.opC"([[OPA]])
// CHECK:    tf_device.return [[OPA]]
// CHECK:    "tf.opD"([[CLUSTER]])

// Preceding user is using resource updated by a nested op.
!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>
// CHECK-LABEL: func @cluster_nested_op_updating_resource
func.func @cluster_nested_op_updating_resource() {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> !tf_res

  "tf.opA"() ({
    "tf.AssignAddVariableOp"(%1, %0) : (!tf_res, tensor<f32>) -> ()
    "tf.terminator"() : () -> ()
  }) { _xla_compile_device_type = "TPU", _replication_info = "foo" } : () -> ()
  "tf.AssignAddVariableOp"(%1, %0) : (!tf_res, tensor<f32>) -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", _replication_info = "foo" , is_stateless = true} : () -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "foo", device = "CPU", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK: [[CONST:%.*]] = "tf.Const"
// CHECK: [[VAR:%.*]] = "tf.VarHandleOp"
// CHECK: "tf_device.cluster"() ({
// CHECK:   "tf.opA"() ({
// CHECK:     "tf.AssignAddVariableOp"([[VAR]], [[CONST]])
// CHECK:    })
// CHECK:    "tf.opB"()
// CHECK:    tf_device.return
// CHECK:  })
// CHECK-SAME: _replication_info = "foo"
// CHECK: "tf.AssignAddVariableOp"([[VAR]], [[CONST]])

// Preceding user is using resource updated by the cluster within a nested op.
// Resource is updated by a cluster op, and opA (not in cluster) is using the
// resource in a nested op. We expect opA to be after the cluster.
// CHECK-LABEL: func @cluster_nested_op_using_resource
func.func @cluster_nested_op_using_resource() {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> !tf_res
  "tf.AssignAddVariableOp"(%1, %0) { _xla_compile_device_type = "TPU", _replication_info = "foo" } : (!tf_res, tensor<f32>) -> ()
  "tf.opA"() ({
    "tf.AssignAddVariableOp"(%1, %0) : (!tf_res, tensor<f32>) -> ()
    "tf.terminator"() : () -> ()
   }) : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", _replication_info = "foo" , is_stateless = true} : () -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "foo", device = "CPU", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK: [[CONST:%.*]] = "tf.Const"
// CHECK: [[VAR:%.*]] = "tf.VarHandleOp"
// CHECK: "tf_device.cluster"() ({
// CHECK:   "tf.AssignAddVariableOp"([[VAR]], [[CONST]])
// CHECK:    "tf.opB"()
// CHECK:    tf_device.return
// CHECK:  })
// CHECK-SAME: _replication_info = "foo"
// CHECK:  "tf.opA"() ({
// CHECK:   "tf.AssignAddVariableOp"([[VAR]], [[CONST]])


// -----


!tf_res = tensor<*x!tf_type.resource<tensor<f32>>>

// Test multiple replicated clusters interleaved and uses resource variables.
// CHECK-LABEL: func @multiple_replicated_interleaved
func.func @multiple_replicated_interleaved(%arg0: !tf_res) {
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "a", num_replicas = 2, topology = "topology"} : () -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "b", num_replicas = 2, topology = "topology"} : () -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "c", num_replicas = 2, topology = "topology"} : () -> ()
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0) : (!tf_res, !tf_res) -> !tf_res
  %1 = "tf.TPUReplicatedInput"(%arg0, %arg0) : (!tf_res, !tf_res) -> !tf_res
  %2 = "tf.TPUReplicatedInput"(%arg0, %arg0) : (!tf_res, !tf_res) -> !tf_res
  %3 = "tf.ReadVariableOp"(%0) {_xla_compile_device_type = "TPU", _replication_info = "a"} : (!tf_res) -> tensor<f32>
  %4 = "tf.ReadVariableOp"(%1) {_xla_compile_device_type = "TPU", _replication_info = "b"} : (!tf_res) -> tensor<f32>
  %5 = "tf.ReadVariableOp"(%2) {_xla_compile_device_type = "TPU", _replication_info = "c"} : (!tf_res) -> tensor<f32>
  %6 = "tf.Identity"(%3) {_xla_compile_device_type = "TPU", _replication_info = "a"} : (tensor<f32>) -> tensor<f32>
  %7 = "tf.Identity"(%4) {_xla_compile_device_type = "TPU", _replication_info = "b"} : (tensor<f32>) -> tensor<f32>
  %8 = "tf.Identity"(%5) {_xla_compile_device_type = "TPU", _replication_info = "c"} : (tensor<f32>) -> tensor<f32>
  %9:2 = "tf.TPUReplicatedOutput"(%6) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %10:2 = "tf.TPUReplicatedOutput"(%7) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %11:2 = "tf.TPUReplicatedOutput"(%8) : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return
}

// CHECK: tf_device.replicate
// CHECK: tf_device.replicate
// CHECK: tf_device.replicate


// -----


// Test cluster that is replicated but has a non TPUReplicatedOutput consumer.
// CHECK-LABEL: func @replicated_non_replicated_output
func.func @replicated_non_replicated_output() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : () -> tensor<i1>
  %1 = "tf.opB"(%0) {is_stateless = true} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}

// CHECK: [[REPLICATE:%.+]]:2 = tf_device.replicate
// CHECK: "tf.opB"([[REPLICATE]]#0)

// -----

// TF produces Identity ops between TPUReplicatedOutput and
// TPUPartitionedOutputV2 ops. This test ensures that they are erased
// and not considered within the clustered computation. It also ensures that
// the expected interleaving pattern is present in the output.

func.func @partitioned_outputs(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) {
  %pi0 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %pi1 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %1 = "tf.TPUReplicatedInput"(%pi0, %pi1) {is_mirrored_variable = true, is_packed = false} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
  %2 = "tf.opA"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", is_stateless = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %3:2 = "tf.TPUReplicatedOutput"(%2) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %4 = "tf.Identity"(%3#0) : (tensor<?xi32>) -> (tensor<?xi32>)
  %5:2 = "tf.TPUPartitionedOutputV2"(%4) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %6 = "tf.Identity"(%3#1) : (tensor<?xi32>) -> (tensor<?xi32>)
  %7:2 = "tf.TPUPartitionedOutputV2"(%6) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, num_cores_per_replica = 2, topology = "topology"} : () -> ()
  func.return %5#0, %5#1, %7#0, %7#1 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>
}

// CHECK: [[REPLICATE:%.+]]:4 = tf_device.replicate
// CHECK: return [[REPLICATE]]#0, [[REPLICATE]]#2, [[REPLICATE]]#1, [[REPLICATE]]#3

// -----

// Ensures that mixed partitioned and replicated outputs
// works in the multi-replica case.
func.func @mixed_partitioned_outputs(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xf32>) {
  %pi0 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %pi1 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %1 = "tf.TPUReplicatedInput"(%pi0, %pi1) {is_mirrored_variable = true, is_packed = false} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
  %2:2 = "tf.opA"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", is_stateless = true} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xf32>)
  %3:2 = "tf.TPUReplicatedOutput"(%2#0) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %5:2 = "tf.TPUPartitionedOutputV2"(%3#0) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %7:2 = "tf.TPUPartitionedOutputV2"(%3#1) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %8:2 = "tf.TPUReplicatedOutput"(%2#1) : (tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
  %9 = "tf.opB"(%5#0, %5#1, %7#0, %7#1) : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
  %10 = "tf.opC"(%8#0, %8#1) : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, num_cores_per_replica = 2, topology = "topology"} : () -> ()
  func.return %9, %10 : tensor<?xi32>, tensor<?xf32>
}

// CHECK: [[REPLICATE:%.+]]:6 = tf_device.replicate
// CHECK: [[OP_B:%.+]] = "tf.opB"([[REPLICATE]]#0, [[REPLICATE]]#2, [[REPLICATE]]#1, [[REPLICATE]]#3)
// CHECK: [[OP_C:%.+]] = "tf.opC"([[REPLICATE]]#4, [[REPLICATE]]#5)

// -----

// For the single replica case:
// - Ensures that Identity ops are ignored.
// - Checks that mixing TPUPartitionedOutputV2 and TPUReplicatedOutput works.

func.func @single_replica_mixed_partitioned_outputs(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xf32>) {
  %0 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %1 = "tf.TPUReplicatedInput"(%0) {is_mirrored_variable = true, is_packed = false} : (tensor<?xi32>) -> (tensor<?xi32>)
  %2:2 = "tf.opA"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", is_stateless = true} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xf32>)
  %3 = "tf.TPUReplicatedOutput"(%2#0) : (tensor<?xi32>) -> (tensor<?xi32>)
  %4 = "tf.Identity"(%3) : (tensor<?xi32>) -> (tensor<?xi32>)
  %5:2 = "tf.TPUPartitionedOutputV2"(%4) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %6 = "tf.TPUReplicatedOutput"(%2#1) : (tensor<?xf32>) -> (tensor<?xf32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, num_cores_per_replica = 2, topology = "topology"} : () -> ()
  func.return %5#0, %5#1, %6 : tensor<?xi32>, tensor<?xi32>, tensor<?xf32>
}

// CHECK: [[CLUSTER:%.+]]:2 = "tf_device.cluster"
// CHECK: [[OUTPUT:%.+]]:2 = "tf.TPUPartitionedOutputV2"([[CLUSTER]]#0)
// CHECK: return [[OUTPUT]]#0, [[OUTPUT]]#1, [[CLUSTER]]#1

// -----

func.func @replica_mismatch(%arg0: tensor<?xi32>) {
  %pi0 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %pi1 = "tf.TPUPartitionedInputV2"(%arg0) {N = 2, partition_dims = [], _XlaSharding = "", is_packed = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %1 = "tf.TPUReplicatedInput"(%pi0, %pi1) {is_mirrored_variable = true, is_packed = false} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
  %2 = "tf.opA"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", is_stateless = true} : (tensor<?xi32>) -> (tensor<?xi32>)
  %3:2 = "tf.TPUReplicatedOutput"(%2) : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %4 = "tf.Identity"(%3#0) : (tensor<?xi32>) -> (tensor<?xi32>)
  // expected-error@+1 {{expected zero or 2 'TPUPartitionedOutput' op(s), instead got 1}}
  %5:2 = "tf.TPUPartitionedOutputV2"(%4) {_XlaSharding = "", partition_dims = []} : (tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, num_cores_per_replica = 2, topology = "topology"} : () -> ()
  func.return
}

// -----


// Test cluster with missing `num_replicas` attribute.
func.func @missing_num_replicas() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : () -> tensor<i1>
  // expected-error@+1 {{'tf.TPUReplicateMetadata' op requires attribute 'num_replicas'}}
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", topology = "topology"} : () -> ()
  func.return
}


// -----


// Test cluster with bad `num_replicas` attribute.
func.func @bad_num_replicas() {
  // expected-error@+1 {{requires 'num_replicas' int attribute to be at least 1}}
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 0, topology = "topology"} : () -> ()
  func.return
}

// -----

// Test cluster with bad `num_cores_per_replica` attribute.
!rtype = tensor<!tf_type.resource<tensor<10x3xf32>>>
func.func @replication_with_model_parallelism(%arg0: !rtype, %arg1: !rtype, %arg2: !rtype, %arg3: !rtype) -> (tensor<10x3xf32>) {
  %2 = "tf.TPUReplicatedInput"(%arg0, %arg2) : (!rtype, !rtype) -> !rtype
  %3 = "tf.TPUReplicatedInput"(%arg1, %arg3) : (!rtype, !rtype) -> !rtype
  // expected-error@+1 {{'tf.TPUPartitionedInputV2' op requires 4 operands but found 2}}
  %4 = "tf.TPUPartitionedInputV2"(%2, %3) {_XlaSharding = "", device = "", partition_dims = []} : (!rtype, !rtype) -> !rtype
  %6 = "tf.opC"(%4) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (!rtype) -> tensor<10x3xf32>
  %7:2 = "tf.TPUReplicatedOutput"(%6) : (tensor<10x3xf32>) -> (tensor<10x3xf32>, tensor<10x3xf32>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_cores_per_replica = 4 : i64, num_replicas = 2 : i64, topology = "topology"} : () -> ()
  func.return %7#0 : tensor<10x3xf32>
}

// -----


// Test cluster with TPUReplicatedInput where the number of operands does not
// match associated `num_replicas` attribute.
func.func @mismatched_replicated_input(%arg0: tensor<i1>) {
  // expected-error@+1 {{'tf.TPUReplicatedInput' op requires 2 operands}}
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0, %arg0) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
  %1 = "tf.opA"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : (tensor<i1>) -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}


// -----


// Test cluster with TPUReplicatedOutput where the number of results does not
// match associated `num_replicas` attribute.
func.func @mismatched_replicated_output() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", name = "name", is_stateless = true} : () -> tensor<i1>
  // expected-error@+1 {{'tf.TPUReplicatedOutput' op requires 2 results}}
  %1:3 = "tf.TPUReplicatedOutput"(%0) : (tensor<i1>) -> (tensor<i1>, tensor<i1>, tensor<i1>)
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}


// -----


// Test unused TPUReplicatedInput that has more than one operand.
func.func @leftover_replicated_input(%arg0: tensor<i1>) {
  %0 = "tf.TPUReplicatedInput"(%arg0, %arg0) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return
}


// -----


// Test unused TPUReplicatedOutput that has more than one result.
func.func @leftover_replicated_output(%arg0: tensor<i1>) {
  %0:2 = "tf.TPUReplicatedOutput"(%arg0) : (tensor<i1>) -> (tensor<i1>, tensor<i1>)
  func.return
}


// -----

// CHECK-LABEL: func @cluster_ops_keep_replicated_core_attr
func.func @cluster_ops_keep_replicated_core_attr() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU_REPLICATED_CORE:0", name = "name", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// CHECK:      "tf.opA"
// CHECK-SAME-DAG: name = "name"
// CHECK-SAME-DAG:  device = "/device:TPU_REPLICATED_CORE:0"
// CHECK:      tf_device.return

// -----

func.func @missing_compilation_attribute() {
  // expected-error@+1 {{'tf.opA' op has '_replication_info' attribute but not '_xla_compile_device_type' attribute which is unsupported}}
  %0 = "tf.opA"() { _replication_info = "replicate", device = "/device:TPU_REPLICATED_CORE:0", name = "name", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// -----

func.func @empty_replication_attribute() {
  // expected-error@+1 {{'tf.opA' op has an empty '_replication_info' attribute}}
  %0 = "tf.opA"() { _xla_compile_device_type = "TPU", _replication_info = "", device = "/device:TPU_REPLICATED_CORE:0", name = "name", is_stateless = true} : () -> tensor<i1>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 1, topology = "topology"} : () -> ()
  func.return
}

// -----

func.func @invalid_device_type() {
  // expected-error@+1 {{'tf.opA' op has invalid '_xla_compile_device_type' value 'XPU'}}
  "tf.opA"() { _xla_compile_device_type = "XPU", _replication_info = "replicate", is_stateless = true} : () -> ()
  func.return
}

// -----

// Check non-replicated case, including expected attributes at device cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK:    tf_device.return
// CHECK:  })  {_replication_info = "__no_replication_cluster", _xla_compile_device_type = "TPU", allow_soft_placement = true, device_assignment = [], num_cores_per_replica = 1 : i32, step_marker_location = "", topology = "", use_spmd_for_xla_partitioning = false}
func.func @valid_compilation_cluster_no_replication() {
  "tf.opA"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
  func.return
}

// -----

// Check non-replicated case, empty op device to no device in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: tf_device.return
// CHECK-NOT: device =
// CHECK: return
func.func @valid_compilation_cluster_no_replication_empty_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = ""} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = ""} : () -> ()
  func.return
}


// Check non-replicated case, including expected device attr in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: device = "/device:TPU:1"
func.func @valid_compilation_cluster_no_replication_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/device:TPU:1"} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = "/device:TPU:1"} : () -> ()
  func.return
}

// -----

// Check conflicting device names
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK-NOT: device =
func.func @do_nothing_if_short_names_conflict() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/replica:1/task:2/device:TPU:1"} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = "/replica:3/task:4/device:TPU:1"} : () -> ()
  func.return
}

// -----

// Check non-replicated case, including expected device attr in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: device = "/task:0/device:TPU:1"
func.func @valid_compilation_cluster_no_replication_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/task:0/device:TPU:1"} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = "/device:TPU:1"} : () -> ()
  func.return
}

// -----

// Check non-replicated case, including expected device attr in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: device = "/task:0/device:TPU:1"
func.func @valid_compilation_cluster_no_replication_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/device:TPU:1"} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = "/task:0/device:TPU:1"} : () -> ()
  func.return
}

// -----

// Check non-replicated case, empty op device to no device in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: tf_device.return
// CHECK-NOT: device =
// CHECK: return
func.func @valid_compilation_cluster_no_replication_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/device:CPU:0"} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", device = "/task:0/device:TPU:1"} : () -> ()
  func.return
}

// -----

// Check non-replicated case, empty op device to no device in cluster.
// CHECK: "tf_device.cluster"()
// CHECK:    "tf.opA"()
// CHECK:    "tf.opB"()
// CHECK: tf_device.return
// CHECK-NOT: device =
// CHECK: return
func.func @valid_compilation_cluster_no_replication_op_device() {
  "tf.opA"() { _xla_compile_device_type = "TPU", device = "/device:CPU:0"} : () -> ()
  func.return
}

// -----
// expected-error@+1 {{found different '_xla_compile_device_type' attribute values (GPU,TPU) in same block which is not supported}}
func.func @invalid_compilation_cluster_mixed_device_types() {
  "tf.opA"() { _xla_compile_device_type = "GPU", is_stateless = true} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
  func.return
}

// -----

// expected-error@+1 {{found different '_xla_compile_device_type' attribute values (CPU,GPU) in same block which is not supported}}
func.func @invalid_compilation_replication_cluster_mixed_device_types() {
  "tf.opA"() { _xla_compile_device_type = "CPU", _replication_info = "cluster", is_stateless = true} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "GPU", _replication_info = "cluster", is_stateless = true} : () -> ()
  func.return
}

// -----

// expected-error@+1 {{found mixed replicated and non-replicated compiled ops in same block which is not supported}}
func.func @mixed_replicated_non_replicated_ops() {
  "tf.opA"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
  "tf.opB"() { _xla_compile_device_type = "TPU", _replication_info = "cluster", is_stateless = true} : () -> ()
  func.return
}

// -----

func.func @cyclic_control_dependency_no_replication() {
  "tf.opA"() {_xla_compile_device_type = "TPU"} : () -> ()
  // expected-warning@+1 {{op has cyclic dependency with a compilation cluster}}
  "tf.opB"() : () -> ()
  "tf.opC"() {_xla_compile_device_type = "TPU"} : () -> ()
  func.return
}

// -----

func.func @cyclic_data_dependency_no_replication() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", is_stateless = true} : () -> (tensor<i32>)
  // expected-warning@+2 {{op has cyclic dependency with a compilation cluster}}
  // expected-error@+1 {{operand #0 does not dominate this use}}
  %1 = "tf.opB"(%0) {is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
  // expected-note@+1 {{operand defined here (op in the same block)}}
  "tf.opC"(%1) {_xla_compile_device_type = "TPU", is_stateless = true} : (tensor<i32>) -> ()
  func.return
}

// -----

func.func @cyclic_control_dependency_replication() {
  "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : () -> ()
  // expected-warning@+1 {{op has cyclic dependency with a compilation cluster}}
  "tf.opB"() : () -> ()
  "tf.opC"() {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : () -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}


// -----

func.func @cyclic_data_dependency_replication() {
  %0 = "tf.opA"() {_xla_compile_device_type = "TPU", is_stateless = true} : () -> (tensor<i32>)
  // expected-warning@+2 {{op has cyclic dependency with a compilation cluster}}
  // expected-error@+1 {{operand #0 does not dominate this use}}
  %1 = "tf.opB"(%0) {is_stateless = true} : (tensor<i32>) -> (tensor<i32>)
  // expected-note@+1 {{operand defined here (op in the same block)}}
  "tf.opC"(%1) {_xla_compile_device_type = "TPU", is_stateless = true} : (tensor<i32>) -> ()
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "cluster", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}

// -----

// expected-warning@+1 {{TPUReplicateMetadata for associated '_replication_info' attribute 'cluster' is missing}}
func.func @missing_metadata() {
  "tf.opA"() {_xla_compile_device_type = "TPU", _replication_info = "cluster"} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @const_with_attrs
func.func @const_with_attrs(%arg0: tensor<*xi32>, %arg1: tensor<?xi64>) -> (tensor<?xi32>, tensor<?xi64>) {
  // CHECK: %{{[a-z0-9_]*}} = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %{{[a-z0-9_]*}} = "tf.Reshape"(%arg0
  // CHECK-NEXT: %{{.*}} = "tf_device.cluster"() ({
  %minus_one = "tf.Const"() {_replication_info = "cluster",
                          _xla_compile_device_type = "TPU",
                          value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  "tf.TPUReplicateMetadata"() {_replication_info = "cluster", num_replicas = 1 : i64} : () -> ()

  %1 = "tf.Reshape"(%arg0, %minus_one) : (tensor<*xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2 = "tf.Identity"(%1) {_replication_info = "cluster", _xla_compile_device_type = "TPU"} : (tensor<?xi32>) -> tensor<?xi32>

  %4 = "tf.Reshape"(%arg1, %minus_one) {_replication_info = "cluster", _xla_compile_device_type = "TPU", device = ""} : (tensor<?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %5 = "tf.Identity"(%4) {_replication_info = "cluster", _xla_compile_device_type = "TPU"} : (tensor<?xi64>) -> tensor<?xi64>

  func.return %2, %5 : tensor<?xi32>, tensor<?xi64>
}

// -----

// CHECK-LABEL: func @two_clusters
func.func @two_clusters(%arg0: tensor<*xi32>, %arg1: tensor<?xi64>) -> (tensor<?xi32>, tensor<?xi64>) {
  // CHECK: %{{[a-z0-9_]*}} = "tf.Const"(){{.*}}value = dense<1>
  // CHECK-NEXT: %{{[a-z0-9_]*}} = "tf.Const"(){{.*}}value = dense<2>
  // CHECK-NEXT: %{{[a-z0-9_]*}} = "tf_device.cluster"
  %one = "tf.Const"() {_replication_info = "cluster1",
                       _xla_compile_device_type = "TPU",
                       value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %two = "tf.Const"() {_replication_info = "cluster2",
                       _xla_compile_device_type = "TPU",
                       value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  "tf.TPUReplicateMetadata"() {_replication_info = "cluster1", num_replicas = 1 : i64} : () -> ()
  "tf.TPUReplicateMetadata"() {_replication_info = "cluster2", num_replicas = 1 : i64} : () -> ()

  %1 = "tf.Reshape"(%arg0, %one) {_replication_info = "cluster2", _xla_compile_device_type = "TPU", device = ""} : (tensor<*xi32>, tensor<1xi32>) -> tensor<?xi32>
  %2 = "tf.Identity"(%1) {_replication_info = "cluster2", _xla_compile_device_type = "TPU"} : (tensor<?xi32>) -> tensor<?xi32>

  %3 = "tf.Reshape"(%arg1, %two) {_replication_info = "cluster1", _xla_compile_device_type = "TPU", device = ""} : (tensor<?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %4 = "tf.Identity"(%3) {_replication_info = "cluster1", _xla_compile_device_type = "TPU"} : (tensor<?xi64>) -> tensor<?xi64>

  func.return %2, %4 : tensor<?xi32>, tensor<?xi64>
}

// -----

// Check that there is one replicate argument for each TPUReplicatedInput
// even when there are multiple uses of the TPUReplicatedInput.

// CHECK-LABEL: func @one_arg_per_TRI
func.func @one_arg_per_TRI(%arg0: tensor<i32>, %arg1: tensor<i32>) -> () {
  // CHECK: tf_device.replicate
  // CHECK-SAME: [%arg0, %arg1] as
  // CHECK-NOT: [%arg0, %arg1] as
  %ri = "tf.TPUReplicatedInput"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "tf.opA"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i32>) -> tensor<i32>
  %1 = "tf.opB"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "replicate", is_stateless = true} : (tensor<i32>) -> tensor<i32>
  "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "/device:TPU:0", num_replicas = 2, topology = "topology"} : () -> ()
  func.return
}

// -----

// Check that for non-TPU device types we don't cluster.

// CHECK-LABEL: func @cpu_device
// CHECK-NOT: tf_device.cluster
func.func @cpu_device() {
  "tf.opA"() { _xla_compile_device_type = "CPU"} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @gpu_device
// CHECK-NOT: tf_device.cluster
func.func @gpu_device() {
  "tf.opA"() { _xla_compile_device_type = "GPU"} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @gather_nd
func.func @gather_nd(%arg0: tensor<*x!tf_type.resource<tensor<80xf32>>>,
                     %arg1: tensor<3xf32>) {
  // CHECK: ResourceGatherNd
  // CHECK: tf_device.cluster
  // CHECK: Add
  // CHECK: ResourceGatherNd
  %0 = "tf.Const"() {value = dense<32> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.ResourceGatherNd"(%arg0, %0) {
    Tindices = i32
  } : (tensor<*x!tf_type.resource<tensor<80xf32>>>, tensor<i32>) -> tensor<1x80xf32>
  %2 = "tf.Add"(%1, %1) {
    _xla_compile_device_type = "TPU",
    device = "/task:0/device:TPU:0", dtype = f32
  } : (tensor<1x80xf32>, tensor<1x80xf32>) -> tensor<1x80xf32>
  %3 = "tf.ResourceGatherNd"(%arg0, %0) {
    Tindices = i32
  } : (tensor<*x!tf_type.resource<tensor<80xf32>>>, tensor<i32>) -> tensor<1x80xf32>
  func.return
}
