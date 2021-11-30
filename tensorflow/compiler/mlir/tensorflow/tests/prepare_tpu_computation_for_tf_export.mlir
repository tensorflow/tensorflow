// RUN: tf-opt %s -split-input-file -verify-diagnostics -prepare-tpu-computation-for-tf-export | FileCheck %s

// CHECK-LABEL: @ShardingAttr
func @ShardingAttr(%arg0: tensor<128x10xf32> {mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"}, %arg1: tensor<10x1024xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<128x1024xf32> {mhlo.sharding = ""}) -> (tensor<128x10xf32>, tensor<10x1024xf32>, tensor<128x1024xf32>) {

  // CHECK: %[[SHARDED_ARG0:.*]] = "tf.XlaSharding"(%arg0) {_XlaSharding = "\08\03\1A\02\01\02\22\02\00\01", sharding = "\08\03\1A\02\01\02\22\02\00\01"}
  // CHECK: %[[SHARDED_ARG1:.*]] = "tf.XlaSharding"(%arg1) {_XlaSharding = "\08\01\1A\01\01\22\01\00", sharding = "\08\01\1A\01\01\22\01\00"}

  // CHECK: "tf.Identity"(%[[SHARDED_ARG1]])
  %0 = "tf.Identity"(%arg1) : (tensor<10x1024xf32>) -> tensor<10x1024xf32>

  // CHECK: "tf.Identity"(%arg2)
  %1 = "tf.Identity"(%arg2) : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
  return %arg0, %0, %1 : tensor<128x10xf32>, tensor<10x1024xf32>, tensor<128x1024xf32>
}

// CHECK-LABEL: @RewriteHostComputeMlirOp
func @RewriteHostComputeMlirOp(%arg0: tensor<*xf32>, %arg1: tensor<3x?xf64>) -> (tensor<*xf32>) {

  // CHECK: "tf.XlaHostCompute"(%arg0, %arg1)
  // CHECK-SAME: ancestors = []
  // CHECK-SAME: cost_estimate_ns = 1000000 : i64
  // CHECK-SAME: key = ""
  // CHECK-SAME: recv_key = "host_compute_channel_recv"
  // CHECK-SAME: send_key = "host_compute_channel_send"
  // CHECK-SAME: shape_inference_graph = @host_func
  // CHECK-SAME: shapes = [#tf_type.shape<*>, #tf_type.shape<3x?>]
  // CHECK-SAME: tpu_core = 0 : i64
  // CHECK: func @host_func
  // CHECK: %[[RECV_OUTPUT:[0-9]*]]:2 = "tf._XlaRecvAtHost"
  // CHECK-SAME: key = "host_compute_channel_send"
  // CHECK-SAME: -> (tensor<*xf32>, tensor<3x?xf64>)
  // CHECK: %[[ID_OUTPUT:[0-9]*]] = "tf.Identity"(%[[RECV_OUTPUT]]#0)
  // CHECK: "tf._XlaSendFromHost"
  // CHECK-SAME: [[ID_OUTPUT]]
  // CHECK-SAME: key = "host_compute_channel_recv"

  %0:2 = "tf._XlaHostComputeMlir"(%arg0, %arg1) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0, host_mlir_module = "module  {\0A  func @host_func(%arg0: tensor<*xf32>, %arg1: tensor<3x?xf64>) -> (tensor<*xf32>, tensor<3x?xf64>) {\0A    %0 = \22tf.Identity\22(%arg0) {_xla_outside_compilation = \22cluster1\22} : (tensor<*xf32>) -> tensor<*xf32> \0A    return %0, %arg1 : tensor<*xf32>, tensor<3x?xf64> \0A  } \0A} \0A"} : (tensor<*xf32>, tensor<3x?xf64>) -> (tensor<*xf32>, tensor<3x?xf64>)
  return %0#0 : tensor<*xf32>
}

// CHECK-LABEL: @RewriteSendRecvOps
func @RewriteSendRecvOps() -> () {
  // CHECK: key = "recv_key_htod_0"
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>

  // CHECK: key = "send_key_dtoh_0"
  "tf.XlaSendToHost"(%0) {key = "send_key"} : (tensor<i32>) -> ()

  return
}

// CHECK-LABEL: @CommunicateOpTokenAttrs
func @CommunicateOpTokenAttrs() -> () {
  // CHECK: _xla_original_oc_node_name = [[NODE_NAME1:.*]], _xla_token_input_nodes = ["_xla_token_arg_node"]
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>

  // CHECK: _xla_original_oc_node_name = [[NODE_NAME2:.*]], _xla_token_input_nodes = {{\[}}[[NODE_NAME1]]{{\]}}
  "tf.XlaSendToHost"(%0) {key = "send_key"} : (tensor<i32>) -> ()

  // CHECK: _xla_original_oc_node_name = [[NODE_NAME3:.*]], _xla_token_input_nodes = {{\[}}[[NODE_NAME2]]{{\]}}
  %1 = "tf._XlaHostComputeMlir"(%0) {recv_key = "host_compute_channel_recv1", send_key = "host_compute_channel_send1", tpu_core = 0, host_mlir_module = ""} : (tensor<i32>) -> (tensor<f32>)
  return
}

// CHECK-LABEL: @IfOpTokenAttrs
func @IfOpTokenAttrs(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>) {
  // CHECK: tf.IfRegion
  %0 = "tf.IfRegion"(%arg0) ({
      // CHECK: tf.XlaRecvFromHost
      // CHECK-SAME-DAG: _xla_original_oc_node_name =
      // CHECK-SAME-DAG: _xla_token_input_nodes = ["_xla_token_arg_node"]
      %recv = "tf.XlaRecvFromHost"() {key = "if_op_token_recv_key", shape = #tf_type.shape<>} : () -> tensor<f32>

      // CHECK: tf.Yield
      "tf.Yield"(%recv) : (tensor<f32>) -> ()
    }, {
      // CHECK-NOT: _xla_token_input_nodes
      %add = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>

      // CHECK: tf.Yield
      "tf.Yield"(%add) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>
  // CHECK: _xla_token_input_nodes = ["_xla_token_arg_node"]

  return %0 : tensor<f32>
}

// Verifies that If ops that don't have any communication ops don't have token
// input nodes attribute even if the parent region has token argument.

// CHECK-LABEL: @IfOpWithoutCommunicationOps
func @IfOpWithoutCommunicationOps(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  // CHECK: tf.IfRegion
  %0 = "tf.IfRegion"(%arg0) ({
      %mul = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%mul) : (tensor<f32>) -> ()
    }, {
      %add = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%add) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>
  // CHECK-NOT: _xla_token_input_nodes

  // CHECK: tf.IfRegion
  %1 = "tf.IfRegion"(%arg0) ({
      %recv = "tf.XlaRecvFromHost"() {key = "if_op_token_recv_key", shape = #tf_type.shape<>} : () -> tensor<f32>
      "tf.Yield"(%recv) : (tensor<f32>) -> ()
    }, {
      %add = "tf.Add"(%arg1, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%add) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>
  // CHECK: _xla_token_input_nodes = ["_xla_token_arg_node"]

  return %0, %1 : tensor<f32>, tensor<f32>
}


// Next four functions are used to verify handling of a call chain.

// CHECK-LABEL: func @IdentityFunc
func @IdentityFunc(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT: _xla_token_input_nodes
  %1 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @PartitionedCall3
func @PartitionedCall3(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: _xla_original_oc_node_name = [[NODE_NAME1:.*]], _xla_token_input_nodes = ["_xla_token_arg_node"]
  "tf.XlaSendToHost"(%arg0) {key = "send_key_call3"} : (tensor<i32>) -> ()
  // CHECK: _xla_original_oc_node_name = [[NODE_NAME2:.*]], _xla_token_input_nodes = {{\[}}[[NODE_NAME1]]{{\]}}
  %1 = "tf.XlaRecvFromHost"() {key = "recv_key_call3", shape = #tf_type.shape<>} : () -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @PartitionedCall2
func @PartitionedCall2(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: _xla_original_oc_node_name = [[NODE_NAME1:.*]], _xla_token_input_nodes = ["_xla_token_arg_node"]
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @PartitionedCall3} : (tensor<i32>) -> (tensor<i32>)
  // CHECK-NOT: _xla_token_input_nodes
  %1 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @IdentityFunc} : (tensor<i32>) -> (tensor<i32>)
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @PartitionedCall1
func @PartitionedCall1(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: _xla_original_oc_node_name = [[NODE_NAME1:.*]], _xla_token_input_nodes = ["_xla_token_arg_node"]
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @PartitionedCall2} : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

// -----

func @Callee(%arg0: tensor<i32>) -> tensor<i32> {
  "tf.XlaSendToHost"(%arg0) {key = "send_key_call3"} : (tensor<i32>) -> ()
  %1 = "tf.XlaRecvFromHost"() {key = "recv_key_call3", shape = #tf_type.shape<>} : () -> tensor<i32>
  return %1 : tensor<i32>
}

func @UnsupportedOp(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{does not support subcomputations with tf/xla communication ops}}
  %0 = "tf.CustomTestOp"(%arg0) {config = "", config_proto = "", executor_type = "", f = @Callee} : (tensor<i32>) -> (tensor<i32>)
  return %0 : tensor<i32>
}

