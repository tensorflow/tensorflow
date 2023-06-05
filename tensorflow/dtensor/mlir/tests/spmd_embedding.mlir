// RUN: dtensor-opt %s -split-input-file -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check that embedding configuration is lowered correctly.
// CHECK-LABEL: func @main
func.func @main() -> () {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   %[[CST:.*]] = "tf.Const"
  // CHECK-NEXT:   %[[IsInit_OUT:.*]] = "tf.IsTPUEmbeddingInitialized"
  // CHECK-NEXT:   "tf.If"(%[[IsInit_OUT:.*]], %[[CST:.*]])
  // CHECK-NEXT: tf_device.return
  // CHECK-NEXT: _mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:EPU:0"
  // Check lowered private if-else branch function.
  // CHECK-LABEL: func private @tf.SetEmbeddingConfig_else_func
  // CHECK-NEXT:   tf.ExecuteTPUEmbeddingPartitioner
  // CHECK-NEXT:   tf.ConfigureTPUEmbeddingMemory
  // CHECK-NEXT:   tf.CollateTPUEmbeddingMemory
  // CHECK-NEXT:   tf.ConfigureTPUEmbeddingHost
  // CHECK-NEXT:   tf.ConnectTPUEmbeddingHost
  // CHECK-NEXT:   tf.FinalizeTPUEmbedding
  "tf_device.cluster"() ({
      %cst = "tf.Const"() {_global_shape = [#tf_type.shape<>], _layout = ["sharding_specs: mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1"], value = dense<"\0A\17\0A\03T_0\10\08\18\04*\0C\22\00j\05\0D\00\00\80?\88\01\02\10\01 \01(\02R\03\1A\01\06"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      "tf.SetEmbeddingConfig"(%cst) : (tensor<1x!tf_type.string>) -> ()
      tf_device.return
    }) {_mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:EPU:0"} : () -> ()
    func.return
}

// -----

// Check DTensorLoadTPUEmbeddingParameters is lowered correctly.
// CHECK-LABEL: func @main
func.func @main(
  %arg0: tensor<i32> {tf._global_shape = #tf_type.shape<>},
  %arg1: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg2: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg3: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg4: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg5: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg6: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg7: tensor<!tf_type.resource<tensor<0x0xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<0x0>, tf._global_shape = #tf_type.shape<0x0>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"},
  %arg8: tensor<!tf_type.resource<tensor<8x4xf32>>> {tf._assigned_resource_local_shape = #tf_type.shape<8x4>, tf._global_shape = #tf_type.shape<8x4>, tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", tf._mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"}
) {
  // CHECK:    "tf_device.cluster"
  //
  // CHECK: %0 = "tf.ReadVariableOp"(%arg1)
  // CHECK-SAME: _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"]
  // CHECK: %7 = "tf.ReadVariableOp"(%arg8)
  // CHECK-SAME: _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"]
  // CHECK-NEXT: tf.LoadAllTPUEmbeddingParameters
  // CHECK-NEXT: tf_device.return
  "tf_device.cluster"() ({
      %0 = "tf.ReadVariableOp"(%arg1) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %1 = "tf.ReadVariableOp"(%arg2) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %2 = "tf.ReadVariableOp"(%arg3) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %3 = "tf.ReadVariableOp"(%arg4) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %4 = "tf.ReadVariableOp"(%arg5) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %5 = "tf.ReadVariableOp"(%arg6) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %6 = "tf.ReadVariableOp"(%arg7) {_global_shape = [#tf_type.shape<0x0>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<0x0xf32>>>) -> tensor<0x0xf32>
      %7 = "tf.ReadVariableOp"(%arg8) {_global_shape = [#tf_type.shape<8x4>], _layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], device = ""} : (tensor<!tf_type.resource<tensor<8x4xf32>>>) -> tensor<8x4xf32>
      "tf.LoadAllTPUEmbeddingParameters"(%7, %0, %1, %2, %3, %4, %5, %6) {_layout = [], config = "\0A\1B\0A\05video\10\08\18\04 \01*\0C\22\00j\05\0D\00\00\80?\88\01\02\10\02\18\08 \01(\02", num_shards = 1 : i64, shard_id = 0 : i64} : (tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>) -> ()
      tf_device.return {_layout = []}
    }) {_mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
    func.return
}

// -----

// Check RetrieveTPUEmbeddingParameters is lowered correctly.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> (tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>) {
  // CHECK:    "tf_device.cluster"
  // CHECK-NEXT: "tf.RetrieveAllTPUEmbeddingParameters"
  // CHECK-NEXT:  tf_device.return
  %0:8 = "tf_device.cluster"() ({
      %parameters, %auxiliary1, %auxiliary2, %auxiliary3, %auxiliary4, %auxiliary5, %auxiliary6, %auxiliary7 = "tf.RetrieveAllTPUEmbeddingParameters"() {_layout = ["sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0", "sharding_specs:batch,unsharded, mesh:|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"], config = "\0A\1B\0A\05video\10\08\18\04 \01*\0C\22\00j\05\0D\00\00\80?\88\01\02\10\02\18\08 \01(\02", num_shards = 1 : i64, shard_id = 0 : i64} : () -> (tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>)
      tf_device.return {_layout = []} %parameters, %auxiliary1, %auxiliary2, %auxiliary3, %auxiliary4, %auxiliary5, %auxiliary6, %auxiliary7 : tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>
    }) {_mesh = "|batch=1|0|0|/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>)
    func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : tensor<8x4xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>, tensor<0x0xf32>
}

// -----

// Check DTensorEmbeddingDequeueOp is lowered correctly.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<12x4xf32> {tf._global_shape = #tf_type.shape<12x4>}) {
  // CHECK:    "tf_device.cluster"
  // CHECK-NEXT: "tf.RecvTPUEmbeddingActivations"()
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
      %1 = "tf.DTensorEmbeddingDequeue"() {config = "\0A\17\0A\03T_0\10\08\18\04*\0C\22\00j\05\0D\00\00\80?\88\01\02\10\01 \01(\02R\03\1A\01\06", output_layouts = ["sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"]} : () -> tensor<12x4xf32>
      %2 = "tf.DTensorLayout"(%1) {global_shape = #tf_type.shape<12x4>, layout = #dtensor.layout<sharding_specs:batch,unsharded, mesh:|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1>} : (tensor<12x4xf32>) -> tensor<12x4xf32>
      tf_device.return %2 : tensor<12x4xf32>
    }) {_mesh = "|batch=2|0,1|0,1|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1"} : () -> tensor<12x4xf32>
  func.return %0 : tensor<12x4xf32>
}

// -----

// Check DTensorSendEmbeddingGradientsOp is lowered correctly.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<12x4xf32> {tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=2,x=1|*TPU"}) -> () {
  // CHECK:    "tf_device.cluster"
  // CHECK-NEXT: "tf.SendTPUEmbeddingGradients"
  // CHECK-NEXT: tf_device.return
  "tf_device.cluster"() ({
    "tf.DTensorSendEmbeddingGradients"(%arg0) {config = "", operand_segment_sizes = array<i32: 1, 0>} : (tensor<12x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "mesh:|batch=2,x=1|*TPU"} : () -> ()
  func.return
}
