// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-embedding-v2 -verify-diagnostics -mlir-print-ir-after-all | FileCheck %s --dump-input=fail


// Check main function and table arg attributes annotation.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:[a-z0-9]*]]: tensor<i32>
// CHECK-SAME: %[[ARG1:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._global_shape = #tf_type.shape<8x4>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._tpu_embedding_slot_id = 0 : i64, tf._tpu_embedding_table_id = 0 : i64}
// CHECK-SAME: %[[ARG2:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._global_shape = #tf_type.shape<8x4>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1", tf._tpu_embedding_slot_id = 1 : i64, tf._tpu_embedding_table_id = 0 : i64}
// CHECK-SAME: attributes {tf._tpu_embedding_configuration =
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg2: tensor<4x3xi32> {tf._layout = "sharding_specs:batch, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg3: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}
) -> (tensor<4x3x4xf32>) {
  // Check configuration ops on host level EPU cluster.
  // CHECK:      "tf_device.cluster"()
  // CHECK:        %cst = "tf.Const"()
  // CHECK-NEXT:   "tf.SetEmbeddingConfig"(%cst) : (tensor<1x!tf_type.string>) -> ()
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:EPU:0"
  // Check removal of old epu cluster mesh.
  // CHECK-NOT:  _mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1"
  // Check dequeue op is added on TPU cluster to return results.
  // CHECK:      %0 = "tf_device.cluster"()
  // CHECK:        %1 = "tf.DTensorEmbeddingDequeue"()
  // CHECK-NEXT:   %2 = "tf.DTensorLayout"(%1)
  // CHECK-NEXT:   %[[REDUCTION_INDICES:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[TARGET:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT:   %3 = "tf.Identity"(%2)
  // CHECK-NEXT:   %4 = "tf.SquaredDifference"(%3, %[[TARGET:.*]])
  // CHECK-NEXT:   %[[LOSS:.*]] = "tf.Sum"(%4, %[[REDUCTION_INDICES:.*]])
  // CHECK-NEXT:   %[[GRAD:.*]] = "tf.Mul"(%3, %[[CST:.*]])
  // CHECK-NEXT:   "tf.DTensorSendEmbeddingGradients"(%[[GRAD:.*]])
  // CHECK-NEXT:   tf_device.return
  // CHECK-SAME:   %[[LOSS:.*]] : tensor<f32>
  // CHECK-NEXT: _mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"
   // Check Enqueue op is added on feature's cluster.
  // CHECK:      "tf_device.cluster"()
  // CHECK:        %1 = "tf.DTensorLayout"(%arg2)
  // CHECK-NEXT:   %cst = "tf.Const"() {value = dense<> : tensor<0xf32>} : () -> tensor<0xf32>
  // CHECK-NEXT:   "tf.DTensorEmbeddingEnqueue"(%1, %cst)
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"
  // Check slot variable remains on embedding table cluster.
  // CHECK:      "tf_device.cluster"()
  // CHECK-NEXT:   %1 = "tf.DTensorLayout"(%arg3)
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"
  // Check main function is end after this.
  // CHECK-NEXT: return %0 : tensor<4x3x4xf32>
  %0 = "tf_device.cluster"() ({
    %cst = "tf.Const"() {device = "", value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_0 = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<2.000000e+00> : tensor<4x3x4xf32>} : () -> tensor<4x3x4xf32>
    %1 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1_2", layout = #dtensor.layout<sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>, shape = #tf_type.shape<4x3x4>} : () -> tensor<4x3x4xf32>
    %2 = "tf.Identity"(%1) : (tensor<4x3x4xf32>) -> tensor<4x3x4xf32>
    %3 = "tf.SquaredDifference"(%2, %cst_0) {device = ""} : (tensor<4x3x4xf32>, tensor<f32>) -> tensor<4x3x4xf32>
    %4 = "tf.Sum"(%3, %cst) {device = "", keep_dims = false} : (tensor<4x3x4xf32>, tensor<3xi32>) -> tensor<f32>
    %5 = "tf.Mul"(%2, %cst_1) {device = ""} : (tensor<4x3x4xf32>, tensor<4x3x4xf32>) -> tensor<4x3x4xf32>
    "tf.DTensorSend"(%5) {key = "communication_key_sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_3", target_layout = #dtensor.layout<sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>} : (tensor<4x3x4xf32>) -> ()
    tf_device.return %4 : tensor<f32>
  }) {_mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : () -> tensor<4x3x4xf32>
  "tf_device.cluster"() ({
    %1 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:batch, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_0", layout = #dtensor.layout<sharding_specs:batch, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>, shape = #tf_type.shape<4>} : () -> tensor<4x3xi32>
    %2 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_1", layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>, shape = #tf_type.shape<8x4>} : () -> tensor<8x4xf32>
    %3 = "tf.TPUDenseEmbeddingLookUp"(%1, %2) : (tensor<4x3xi32>, tensor<8x4xf32>) -> tensor<4x3x4xf32>
    "tf.DTensorSend"(%3) {key = "communication_key_sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1_2", target_layout = #dtensor.layout<sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<4x3x4xf32>) -> ()
    %4 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_3", layout = #dtensor.layout<sharding_specs:batch,unsharded,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>, shape = #tf_type.shape<4x3x4>} : () -> tensor<4x3x4xf32>
    %5 = "tf.TPUDenseEmbeddingLookUpGrad"(%4, %1, %2) {device = ""} : (tensor<4x3x4xf32>, tensor<4x3xi32>, tensor<8x4xf32>) -> tensor<8x4xf32>
    "tf.DTensorSend"(%5) {key = "communication_key_sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_4", target_layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<8x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1"} : () -> ()
  "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<4x3>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1>} : (tensor<4x3xi32>) -> tensor<4x3xi32>
    "tf.DTensorSend"(%1) {key = "communication_key_sharding_specs:batch, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_0", target_layout = #dtensor.layout<sharding_specs:batch, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>} : (tensor<4x3xi32>) -> ()
    tf_device.return
  }) {_mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"} : () -> ()
  "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<*x!tf_type.resource<tensor<8x4xf32>>>) -> tensor<*x!tf_type.resource<tensor<8x4xf32>>>
    %2 = "tf.ReadVariableOp"(%1) {device = ""} : (tensor<*x!tf_type.resource<tensor<8x4xf32>>>) -> tensor<8x4xf32>
    %3 = "tf.ReadVariableOp"(%1) {device = ""} : (tensor<*x!tf_type.resource<tensor<8x4xf32>>>) -> tensor<8x4xf32>
    "tf.DTensorSend"(%3) {key = "communication_key_sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1_1", target_layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:EPU:0,/job:localhost/replica:0/task:0/device:EPU:1>} : (tensor<8x4xf32>) -> ()
    %4 = "tf.DTensorRecv"() {key = "communication_key_sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0_4", layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>, shape = #tf_type.shape<8x4>} : () -> tensor<8x4xf32>
    %5 = "tf.DTensorLayout"(%arg3) {global_shape = #tf_type.shape<8x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0>} : (tensor<*x!tf_type.resource<tensor<8x4xf32>>>) -> tensor<*x!tf_type.resource<tensor<8x4xf32>>>
    %6 = "tf.ReadVariableOp"(%5) {device = ""} : (tensor<*x!tf_type.resource<tensor<8x4xf32>>>) -> tensor<8x4xf32>
    "tf.ApplyEmbeddingOptimizerV2"(%4, %2, %6) {device = "", optimization_parameters = "\22\00j\05\0D\CD\CC\CC=\88\01\02"} : (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  func.return %0 : tensor<4x3x4xf32>
}

// -----

// Non epu embedding look up op.
func.func @main(%arg0: tensor<8x4xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU", tf._mesh = "CPU|x=1|*CPU"},
  %arg1: tensor<4xi32>{tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU", tf._mesh = "CPU|x=1|*CPU"}) -> () {
  "tf_device.cluster"() ({
    // expected-error @+1 {{'tf.TPUDenseEmbeddingLookUp' op Expected embedding lookup op defined on EPU cluster but got : CPU}}
    "tf.TPUDenseEmbeddingLookUp"(%arg1, %arg0) : (tensor<4xi32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    tf_device.return
  }) {_mesh = "CPU|x=1|*CPU"} : () -> ()
  func.return
}

// -----

// Failed to get cluster device of embedding lookup op.
func.func @main(%arg0: tensor<8x4xf32> {tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU", tf._mesh = "CPU|x=1|*CPU"},
  %arg1: tensor<4xi32>{tf._layout = "sharding_specs:x, mesh:CPU|x=1|*CPU", tf._mesh = "CPU|x=1|*CPU"}) -> () {
  "tf_device.cluster"() ({
    // expected-error @+1 {{'tf.TPUDenseEmbeddingLookUp' op Failed to get device type of cluster has embedding look op, got error: Cluster Mesh is not found.}}
    "tf.TPUDenseEmbeddingLookUp"(%arg1, %arg0) : (tensor<4xi32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    tf_device.return
  }) {_mesh = ""} : () -> ()
  func.return
}
