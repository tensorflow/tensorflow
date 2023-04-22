// RUN: tf-opt -split-input-file -verify-diagnostics -xla-legalize-tf-communication %s | FileCheck %s

// Test legalization of `tf._XlaHostComputeMlir` expands into individual
// `mhlo.send` per operand and `mhlo.recv` per result. Channel Id's are uniquely
// assigned per mhlo communcation op, and frontend attributes (modified keys)
// and op shardings (based on `tpu_core`) are added. Sink tokens are created
// if there are more than one operand or more than one result.
//
// The following op sharding is used:
// Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0
// Serialized string:
//   "\08\01\1A\01\01\22\01\00"

// CHECK-LABEL: func @host_compute
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>, [[ARG1:%.*]]: tensor<i64>)
func @host_compute(%arg0: tensor<i32>, %arg1: tensor<i64>) -> (tensor<f32>, tensor<f64>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_0"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_ARG1_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s64", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_1"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i64>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_SINK_TOKEN:%.*]] = "mhlo.after_all"([[SEND_ARG0_TOKEN]], [[SEND_ARG1_TOKEN]])

  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND_SINK_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 3 : i64, type = 3 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "host_compute_channel_recv_htod_0"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (!mhlo.token) -> tuple<tensor<f32>, !mhlo.token>

  // CHECK:      [[RECV_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tuple<tensor<f32>, !mhlo.token>) -> tensor<f32>

  // CHECK:      [[RECV_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tuple<tensor<f32>, !mhlo.token>) -> !mhlo.token

  // CHECK:      [[RECV_RETVAL1_TUPLE:%.*]] = "mhlo.recv"([[SEND_SINK_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 4 : i64, type = 3 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f64", _xla_host_transfer_rendezvous = "host_compute_channel_recv_htod_1"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (!mhlo.token) -> tuple<tensor<f64>, !mhlo.token>

  // CHECK:      [[RECV_RETVAL1_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL1_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tuple<tensor<f64>, !mhlo.token>) -> tensor<f64>

  // CHECK:      [[RECV_RETVAL1_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL1_TUPLE]])
  // CHECK-SAME: index = 1
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tuple<tensor<f64>, !mhlo.token>) -> !mhlo.token

  // CHECK:      [[RECV_SINK_TOKEN:%.*]] = "mhlo.after_all"([[RECV_RETVAL0_TOKEN]], [[RECV_RETVAL1_TOKEN]])
  %0:2 = "tf._XlaHostComputeMlir"(%arg0, %arg1) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>, tensor<i64>) -> (tensor<f32>, tensor<f64>)

  // CHECK:      return [[RECV_RETVAL0_VAL]], [[RECV_RETVAL1_VAL]] : tensor<f32>, tensor<f64>
  return %0#0, %0#1 : tensor<f32>, tensor<f64>
}

// -----

// Tests `tf._XlaHostComputeMlir` with `tpu_core` assigns the correct op
// sharding.
//
// The following op sharding is used:
// Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 1
// Serialized string:
//   "\08\01\1A\01\01\22\01\01"

// CHECK-LABEL: func @host_compute_sharding
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @host_compute_sharding(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      "mhlo.send"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  // CHECK:      "mhlo.recv"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  // CHECK:      "mhlo.get_tuple_element"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  // CHECK:      "mhlo.get_tuple_element"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 1 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Tests `tf._XlaHostComputeMlir` with no operands simply forwards the input
// token to its generated `mhlo.recv`.

// CHECK-LABEL: func @host_compute_no_operands_one_result
func @host_compute_no_operands_one_result() {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK-NOT:  "mhlo.send"
  // CHECK-NOT:  "mhlo.after_all"
  // CHECK:      "mhlo.recv"([[INIT_TOKEN]])
  %0 = "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> tensor<i32>
  return
}

// -----

// Tests `tf._XlaHostComputeMlir` with no results simply forwards its token from
// the generated `mhlo.send`.

// CHECK-LABEL: func @host_compute_one_operand_no_results
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @host_compute_one_operand_no_results(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> ()

  // CHECK:      "mhlo.recv"([[SEND_TOKEN]])
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>
  return
}

// -----

// Tests `tf._XlaHostComputeMlir` with one operand and one result does not
// create any `mhlo.after_all` ops.

// CHECK-LABEL: func @host_compute_single_operand_result
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @host_compute_single_operand_result(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  // CHECK:      "mhlo.recv"([[SEND_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>
  return
}

// -----

// Test legalization of `tf.XlaSendToHost` expands into a `mhlo.send` op.

// CHECK-LABEL: func @send_to_host
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @send_to_host(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send_key_dtoh_0"}
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> !mhlo.token
  "tf.XlaSendToHost"(%arg0) {key = "send_key"} : (tensor<i32>) -> ()
  return
}

// -----

// Test legalization of `tf.XlaRecvFromHost` expands into a `mhlo.recv` op.

// CHECK-LABEL: func @recv_from_host
func @recv_from_host() -> tensor<i32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]] = "mhlo.recv"([[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 3 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv_key_htod_0"}
  // CHECK-SAME: (!mhlo.token) -> tuple<tensor<i32>, !mhlo.token>


  // CHECK:      [[RECV_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK-SAME: (tuple<tensor<i32>, !mhlo.token>) -> tensor<i32>

  // CHECK:      [[RECV_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 1
  // CHECK-SAME: (tuple<tensor<i32>, !mhlo.token>) -> !mhlo.token
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>

  // CHECK:      return [[RECV_VAL]] : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Test legalization of multiple TF/XLA communication ops are sequenced with
// their generated tokens. Channel Id's are also uniquely assigned.

// CHECK-LABEL: func @multiple_consecutive_ops
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @multiple_consecutive_ops(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND0_ARG0_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send0_dtoh_0"}

  // CHECK:      [[RECV0_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND0_ARG0_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv0_htod_0"}

  // CHECK:      [[RECV0_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV0_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0

  // CHECK:      [[RECV0_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV0_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv0", send_key = "send0", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      [[SEND1_ARG0_TOKEN:%.*]] = "mhlo.send"([[RECV0_RETVAL0_VAL]], [[RECV0_RETVAL0_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 3 : i64, type = 2 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send1_dtoh_0"}

  // CHECK:      [[RECV1_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND1_ARG0_TOKEN]])
  // CHECK-SAME: channel_handle = {handle = 4 : i64, type = 3 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv1_htod_0"}

  // CHECK:      [[RECV1_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV1_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0

  // CHECK:      [[RECV1_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV1_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %1 = "tf._XlaHostComputeMlir"(%0) {recv_key = "recv1", send_key = "send1", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV1_RETVAL0_VAL]] : tensor<i32>
  return %1 : tensor<i32>
}

// -----

// Test private function with TF/XLA communication op being called by another
// function gets rewritten with an extra token argument and an extra token
// result, and the caller passes in a token. The top level function not called
// (or public) will be updated to create a token.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL:%.*]]:2 = call @callee([[MAIN_ARG0]], [[MAIN_TOKEN]])
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> (tensor<i32>, !mhlo.token)
  %0 = call @callee(%arg0) : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[CALL]]#0
  return %0 : tensor<i32>
}

// CHECK: func private @callee([[CALLEE_ARG0:%.*]]: tensor<i32>, [[CALLEE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
func private @callee(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_ARG0]], [[CALLEE_ARG1]])
  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND_ARG0_TOKEN]])
  // CHECK:      [[RECV_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      [[RECV_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV_RETVAL0_VAL]], [[RECV_RETVAL0_TOKEN]]
  return %0 : tensor<i32>
}

// -----

// Test public function with TF/XLA communication op being called by another
// function. The original public function will be modified to create a token,
// while the function is cloned and rewritten with an extra token argument and
// an extra token result. All callers to the original function are updated to
// point to the cloned function and the function the caller is in is updated to
// pass a token or create a token.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL:%.*]]:2 = call [[CALLEE_CLONE:@.*]]([[MAIN_ARG0]], [[MAIN_TOKEN]])
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> (tensor<i32>, !mhlo.token)
  %0 = call @callee(%arg0) : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[CALL]]#0 : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK: func @callee([[CALLEE_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func @callee(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[CALLEE_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_ARG0]], [[CALLEE_TOKEN]])
  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND_ARG0_TOKEN]])
  // CHECK:      [[RECV_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      [[RECV_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV_RETVAL0_VAL]]
  return %0 : tensor<i32>
}

// CHECK: func private [[CALLEE_CLONE]]([[CALLEE_CLONE_ARG0:%.*]]: tensor<i32>, [[CALLEE_CLONE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
// CHECK-NOT:  "mhlo.create_token"

// CHECK:      [[CLONE_SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_CLONE_ARG0]], [[CALLEE_CLONE_ARG1]])
// CHECK:      [[CLONE_RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[CLONE_SEND_ARG0_TOKEN]])
// CHECK:      [[CLONE_RECV_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[CLONE_RECV_RETVAL0_TUPLE]])
// CHECK-SAME: index = 0
// CHECK:      [[CLONE_RECV_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[CLONE_RECV_RETVAL0_TUPLE]])
// CHECK-SAME: index = 1

// CHECK:      return [[CLONE_RECV_RETVAL0_VAL]], [[CLONE_RECV_RETVAL0_TOKEN]]

// -----

// Tests generated tokens are passed into a function call that also has TF/XLA
// communication ops.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>)
func @main(%arg0: tensor<i32>) {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[MAIN_SEND0_TOKEN:%.*]] = "mhlo.send"([[MAIN_ARG0]], [[MAIN_TOKEN]])
  "tf.XlaSendToHost"(%arg0) {key = "send0"} : (tensor<i32>) -> ()

  // CHECK:      [[CALL_TOKEN:%.*]] = call @callee([[MAIN_SEND0_TOKEN]])
  // CHECK-SAME: (!mhlo.token) -> !mhlo.token
  call @callee() : () -> ()

  // CHECK:      [[MAIN_SEND2_TOKEN:%.*]] = "mhlo.send"([[MAIN_ARG0]], [[CALL_TOKEN]])
  "tf.XlaSendToHost"(%arg0) {key = "send2"} : (tensor<i32>) -> ()
  return
}

// CHECK: func private @callee([[CALLEE_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func private @callee() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[ZERO:%.*]] = mhlo.constant dense<0>
  %0 = mhlo.constant dense<0> : tensor<i32>

  // CHECK:      [[CALLEE_SEND_TOKEN:%.*]] = "mhlo.send"([[ZERO]], [[CALLEE_ARG0]])
  "tf.XlaSendToHost"(%0) {key = "send1"} : (tensor<i32>) -> ()

  // CHECK:      return [[CALLEE_SEND_TOKEN]]
  return
}

// -----

// Test only the top level function generates a token.

// CHECK: func private @callee0()
func private @callee0() {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      call @callee1([[INIT_TOKEN]])
  call @callee1() : () -> ()
  return
}

// CHECK: func private @callee1([[CALLEE1_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func private @callee1() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[CALL_2:%.*]] = call @callee2([[CALLEE1_ARG0]])
  call @callee2() : () -> ()

  // CHECK:      return [[CALL_2]]
  return
}

// CHECK: func private @callee2([[CALLEE2_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func private @callee2() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]] = "mhlo.recv"([[CALLEE2_ARG0]])
  // CHECK:      [[RECV_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      [[RECV_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>

  // CHECK:      return [[RECV_TOKEN]]
  return
}

// -----

// Test cloned function rewrite also checks transitive function calls to
// TF/XLA communication ops.

// CHECK: func @callee3()
func @callee3() {
  // CHECK:      [[CALLEE3_INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      call @callee4{{.+}}([[CALLEE3_INIT_TOKEN]])
  call @callee4() : () -> ()
  return
}

// CHECK: func @callee4()
func @callee4() {
  // CHECK:      [[CALLEE4_INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL_5:%.*]] = call @callee5([[CALLEE4_INIT_TOKEN]])
  call @callee5() : () -> ()

  // CHECK:      return
  return
}

// CHECK: func private @callee5([[CALLEE5_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func private @callee5() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]] = "mhlo.recv"([[CALLEE5_ARG0]])
  // CHECK:      [[RECV_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      [[RECV_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>

  // CHECK:      return [[RECV_TOKEN]]
  return
}

// CHECK: func private @callee4{{.+}}([[CALLEE4_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
// CHECK-NOT:  "mhlo.create_token"
// CHECK:      [[CALL_5:%.*]] = call @callee5([[CALLEE4_ARG0]])
// CHECK:      return [[CALL_5]]

// -----

// Tests `mhlo.if` with branches populated with TF/XLA communication ops.

// CHECK-LABEL: func @if_both_branches
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func @if_both_branches(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[TRUE_TUPLE:%.*]] = "mhlo.tuple"([[ARG1]], [[INIT_TOKEN]])
  // CHECK: [[FALSE_TUPLE:%.*]] = "mhlo.tuple"([[ARG2]], [[INIT_TOKEN]])

  // CHECK: [[IF_TUPLE:%.*]] = "mhlo.if"([[ARG0]], [[TRUE_TUPLE]], [[FALSE_TUPLE]])
  %0 = "mhlo.if"(%arg0, %arg1, %arg2) ( {
  // CHECK: ^bb0([[TRUE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[TRUE_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[TRUE_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 1

    // CHECK:      [[TRUE_SEND_TOKEN:%.*]] = "mhlo.send"([[TRUE_REGION_ARG_VALUE]], [[TRUE_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_true_dtoh_0"}

    // CHECK:      [[TRUE_RECV_TUPLE:%.*]] = "mhlo.recv"([[TRUE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_true_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg3) {recv_key = "recv_if_true", send_key = "send_if_true", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[TRUE_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[TRUE_RECV_TUPLE]]) {index = 1
    // CHECK:      [[TRUE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[TRUE_GET_TUPLE_ELEMENT0]], [[TRUE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[TRUE_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
  // CHECK: ^bb0([[FALSE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[FALSE_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[FALSE_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 1

    // CHECK:      [[FALSE_SEND_TOKEN:%.*]] = "mhlo.send"([[FALSE_REGION_ARG_VALUE]], [[FALSE_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 3 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_false_dtoh_0"}

    // CHECK:      [[FALSE_RECV_TUPLE:%.*]] = "mhlo.recv"([[FALSE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 4 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_false_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg3) {recv_key = "recv_if_false", send_key = "send_if_false", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[FALSE_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[FALSE_RECV_TUPLE]]) {index = 1
    // CHECK:      [[FALSE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[FALSE_GET_TUPLE_ELEMENT0]], [[FALSE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[FALSE_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>, tuple<tensor<f32>, !mhlo.token>, tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK:      [[IF_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[IF_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with only the `true` branch populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @if_true_branch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func @if_true_branch(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[TRUE_TUPLE:%.*]] = "mhlo.tuple"([[ARG1]], [[INIT_TOKEN]])
  // CHECK: [[FALSE_TUPLE:%.*]] = "mhlo.tuple"([[ARG2]], [[INIT_TOKEN]])

  // CHECK: [[IF_TUPLE:%.*]] = "mhlo.if"([[ARG0]], [[TRUE_TUPLE]], [[FALSE_TUPLE]])
  %0 = "mhlo.if"(%arg0, %arg1, %arg2) ( {
  // CHECK: ^bb0([[TRUE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[TRUE_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[TRUE_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 1

    // CHECK:      [[TRUE_SEND_TOKEN:%.*]] = "mhlo.send"([[TRUE_REGION_ARG_VALUE]], [[TRUE_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_true_dtoh_0"}

    // CHECK:      [[TRUE_RECV_TUPLE:%.*]] = "mhlo.recv"([[TRUE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_true_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg3) {recv_key = "recv_if_true", send_key = "send_if_true", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[TRUE_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[TRUE_RECV_TUPLE]]) {index = 1
    // CHECK:      [[TRUE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[TRUE_GET_TUPLE_ELEMENT0]], [[TRUE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[TRUE_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
  // CHECK: ^bb0([[FALSE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 1
    // CHECK:      [[FALSE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[FALSE_GET_TUPLE_ELEMENT0]], [[FALSE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[FALSE_RETURN_TUPLE]])
    "mhlo.return"(%arg3) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>, tuple<tensor<f32>, !mhlo.token>, tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK:      [[IF_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[IF_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with only the `false` branch populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @if_false_branch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func @if_false_branch(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[TRUE_TUPLE:%.*]] = "mhlo.tuple"([[ARG1]], [[INIT_TOKEN]])
  // CHECK: [[FALSE_TUPLE:%.*]] = "mhlo.tuple"([[ARG2]], [[INIT_TOKEN]])

  // CHECK: [[IF_TUPLE:%.*]] = "mhlo.if"([[ARG0]], [[TRUE_TUPLE]], [[FALSE_TUPLE]])
  %0 = "mhlo.if"(%arg0, %arg1, %arg2) ( {
  // CHECK: ^bb0([[TRUE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[TRUE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 1
    // CHECK:      [[TRUE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[TRUE_GET_TUPLE_ELEMENT0]], [[TRUE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[TRUE_RETURN_TUPLE]])
    "mhlo.return"(%arg3) : (tensor<f32>) -> ()
  },  {
  // CHECK: ^bb0([[FALSE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg3: tensor<f32>):
    // CHECK-DAG:  [[FALSE_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[FALSE_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[FALSE_REGION_ARG]]) {index = 1

    // CHECK:      [[FALSE_SEND_TOKEN:%.*]] = "mhlo.send"([[FALSE_REGION_ARG_VALUE]], [[FALSE_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_false_dtoh_0"}

    // CHECK:      [[FALSE_RECV_TUPLE:%.*]] = "mhlo.recv"([[FALSE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_false_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg3) {recv_key = "recv_if_false", send_key = "send_if_false", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[FALSE_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[FALSE_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[FALSE_RECV_TUPLE]]) {index = 1
    // CHECK:      [[FALSE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[FALSE_GET_TUPLE_ELEMENT0]], [[FALSE_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[FALSE_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>, tuple<tensor<f32>, !mhlo.token>, tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK:      [[IF_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[IF_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with tuple arg from a `mhlo.tuple` only used by `mhlo.if` is
// replaced.

// CHECK-LABEL: func @if_replace_tuple_arg
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func @if_replace_tuple_arg(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK-NOT:  "mhlo.tuple"([[ARG1]], [[ARG2]])
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK:      [[IF_ARG_TUPLE:%.*]] = "mhlo.tuple"([[ARG1]], [[ARG2]], [[INIT_TOKEN]])
  %0 = "mhlo.tuple"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>

  // CHECK: [[IF_TUPLE:%.*]] = "mhlo.if"([[ARG0]], [[IF_ARG_TUPLE]], [[IF_ARG_TUPLE]])
  %1 = "mhlo.if"(%arg0, %0, %0) ( {
  ^bb0(%arg3: tuple<tensor<f32>, tensor<f32>>):
    %2 = "mhlo.get_tuple_element"(%arg3) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    "tf.XlaSendToHost"(%2) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg3: tuple<tensor<f32>, tensor<f32>>):
    %2 = "mhlo.get_tuple_element"(%arg3) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

// Tests `mhlo.if` with tuple arg not from a `mhlo.tuple` is unpacked.

// CHECK-LABEL: func @if_unpack_tuple_arg
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tuple<tensor<f32>, tensor<f32>>)
func @if_unpack_tuple_arg(%arg0: tensor<i1>, %arg1: tuple<tensor<f32>, tensor<f32>>) -> tensor<f32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK-DAG:  [[IF_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[ARG1]]) {index = 0
  // CHECK-DAG:  [[IF_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[ARG1]]) {index = 1
  // CHECK:      [[IF_ARG_TUPLE:%.*]] = "mhlo.tuple"([[IF_ARG_ELEMENT0]], [[IF_ARG_ELEMENT1]], [[INIT_TOKEN]])

  // CHECK:      [[IF_TUPLE:%.*]] = "mhlo.if"([[ARG0]], [[IF_ARG_TUPLE]], [[IF_ARG_TUPLE]])
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>):
    %1 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    "tf.XlaSendToHost"(%1) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>):
    %1 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` tuple result is extended with a `mhlo.token`.

// CHECK-LABEL: func @if_extend_tuple_result
func @if_extend_tuple_result(%arg0: tensor<i1>, %arg1: tuple<tensor<f32>, tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>> {
  // CHECK:      [[IF_TUPLE:%.*]] = "mhlo.if"
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>):
    %1 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
    "tf.XlaSendToHost"(%1) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%arg2) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  },  {
  ^bb0(%arg2: tuple<tensor<f32>, tensor<f32>>):
    "mhlo.return"(%arg2) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  // CHECK:      (tensor<i1>, tuple<tensor<f32>, tensor<f32>, !mhlo.token>, tuple<tensor<f32>, tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, tensor<f32>, !mhlo.token>
  }) : (tensor<i1>, tuple<tensor<f32>, tensor<f32>>, tuple<tensor<f32>, tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>>

  // CHECK-DAG:  [[IF_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]]) {index = 0
  // CHECK-DAG:  [[IF_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]]) {index = 1
  // CHECK:      [[IF_SUBTUPLE_RESULT:%.*]] = "mhlo.tuple"([[IF_TUPLE_ELEMENT0]], [[IF_TUPLE_ELEMENT1]])
  // CHECK:      return [[IF_SUBTUPLE_RESULT]]
  return %0 : tuple<tensor<f32>, tensor<f32>>
}

// -----

// Tests nested `mhlo.if` containing TF/XLA communication ops.

// CHECK-LABEL: func @if_nested
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>)
func @if_nested(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK:      [[OUTER_IF_ARG_TUPLE:%.*]] = "mhlo.tuple"([[ARG1]], [[INIT_TOKEN]])

  // CHECK:      "mhlo.if"([[ARG0]], [[OUTER_IF_ARG_TUPLE]], [[OUTER_IF_ARG_TUPLE]])
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  // CHECK-NEXT: ^bb0([[OUTER_IF_TRUE_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg2: tensor<f32>):
    // CHECK-DAG:  [[OUTER_IF_TRUE_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[OUTER_IF_TRUE_ARG]]) {index = 0
    // CHECK-DAG:  [[OUTER_IF_TRUE_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[OUTER_IF_TRUE_ARG]]) {index = 1
    // CHECK:      [[INNER_IF_ARG_TUPLE:%.*]] = "mhlo.tuple"([[OUTER_IF_TRUE_ARG_ELEMENT0]], [[OUTER_IF_TRUE_ARG_ELEMENT1]])

    %1 = mhlo.constant dense<false> : tensor<i1>

    // CHECK:      [[INNER_IF_TUPLE:%.*]] = "mhlo.if"({{%.*}}, [[INNER_IF_ARG_TUPLE]], [[INNER_IF_ARG_TUPLE]])
    %2 = "mhlo.if"(%1, %arg2, %arg2) ( {
    // CHECK-NEXT: ^bb0([[INNER_IF_TRUE_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
    ^bb0(%arg3: tensor<f32>):
      // CHECK-DAG:  [[INNER_IF_TRUE_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[INNER_IF_TRUE_ARG]]) {index = 0
      // CHECK-DAG:  [[INNER_IF_TRUE_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[INNER_IF_TRUE_ARG]]) {index = 1

      // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[INNER_IF_TRUE_ARG_ELEMENT0]], [[INNER_IF_TRUE_ARG_ELEMENT1]])
      "tf.XlaSendToHost"(%arg3) {key = "send_key"} : (tensor<f32>) -> ()

      // CHECK:      [[INNER_IF_TRUE_RESULT:%.*]] = "mhlo.tuple"([[INNER_IF_TRUE_ARG_ELEMENT0]], [[SEND_TOKEN]])
      // CHECK:      "mhlo.return"([[INNER_IF_TRUE_RESULT]])
      "mhlo.return"(%arg3) : (tensor<f32>) -> ()

    // CHECK-NEXT: },  {
    },  {

    // CHECK-NEXT: ^bb0([[INNER_IF_FALSE_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
    ^bb0(%arg3: tensor<f32>):
      // CHECK-DAG:  [[INNER_IF_FALSE_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[INNER_IF_FALSE_ARG]]) {index = 0
      // CHECK-DAG:  [[INNER_IF_FALSE_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[INNER_IF_FALSE_ARG]]) {index = 1
      // CHECK:      [[INNER_IF_FALSE_RESULT:%.*]] = "mhlo.tuple"([[INNER_IF_FALSE_ARG_ELEMENT0]], [[INNER_IF_FALSE_ARG_ELEMENT1]])
      // CHECK:      "mhlo.return"([[INNER_IF_FALSE_RESULT]])
      "mhlo.return"(%arg3) : (tensor<f32>) -> ()
    // CHECK-NEXT: (tensor<i1>, tuple<tensor<f32>, !mhlo.token>, tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[INNER_IF_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[INNER_IF_TUPLE]]) {index = 1
    // CHECK:      [[OUTER_IF_TRUE_RESULT:%.*]] = "mhlo.tuple"([[OUTER_IF_TRUE_ARG_ELEMENT0]], [[INNER_IF_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[OUTER_IF_TRUE_RESULT]])
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()

  // CHECK-NEXT: },  {
  },  {

  // CHECK-NEXT: ^bb0([[OUTER_IF_FALSE_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg2: tensor<f32>):
    // CHECK-DAG:  [[OUTER_IF_FALSE_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[OUTER_IF_FALSE_ARG]]) {index = 0
    // CHECK-DAG:  [[OUTER_IF_FALSE_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[OUTER_IF_FALSE_ARG]]) {index = 1
    // CHECK:      [[OUTER_IF_FALSE_RESULT:%.*]] = "mhlo.tuple"([[OUTER_IF_FALSE_ARG_ELEMENT0]], [[OUTER_IF_FALSE_ARG_ELEMENT1]])
    // CHECK:      "mhlo.return"([[OUTER_IF_FALSE_RESULT]])
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  // CHECK-NEXT: (tensor<i1>, tuple<tensor<f32>, !mhlo.token>, tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` containing a function call to TF/XLA communication ops.

// CHECK-LABEL: func @if_function_call
func @if_function_call(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.if"
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  // CHECK: ^bb0([[TRUE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg2: tensor<f32>):
    // CHECK-DAG:  [[TRUE_REGION_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[TRUE_REGION_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 1
    // CHECK:      [[CALL_TOKEN:%.*]] = call @callee([[TRUE_REGION_ARG_ELEMENT0]], [[TRUE_REGION_ARG_ELEMENT1]])
    call @callee(%arg2) : (tensor<f32>) -> ()

    // CHECK:      [[TRUE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[TRUE_REGION_ARG_ELEMENT0]], [[CALL_TOKEN]])
    // CHECK:      "mhlo.return"([[TRUE_RETURN_TUPLE]])
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg2: tensor<f32>):
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func private @callee
// CHECK-SAME:  ([[CALLEE_ARG0:%.*]]: tensor<f32>, [[CALLEE_ARG1:%.*]]: !mhlo.token) -> !mhlo.token
func private @callee(%arg0: tensor<f32>) {
  // CHECK: [[SEND_TOKEN:%.*]] = "mhlo.send"
  "tf.XlaSendToHost"(%arg0) {key = "send_key"} : (tensor<f32>) -> ()

  // CHECK: return [[SEND_TOKEN]]
  return
}

// -----

// Tests `mhlo.if` containing multiple TF/XLA communication ops.

// CHECK-LABEL: func @if_region_multiple_ops
func @if_region_multiple_ops(%arg0: tensor<i1>, %arg1: tensor<f32>) {
  // CHECK: "mhlo.if"
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  // CHECK: ^bb0([[TRUE_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg2: tensor<f32>):
    // CHECK: [[TRUE_REGION_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 0
    // CHECK: [[TRUE_REGION_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[TRUE_REGION_ARG]]) {index = 1

    // CHECK: [[SEND0_TOKEN:%.*]] = "mhlo.send"([[TRUE_REGION_ARG_ELEMENT0]], [[TRUE_REGION_ARG_ELEMENT1]])
    "tf.XlaSendToHost"(%arg2) {key = "send_key0"} : (tensor<f32>) -> ()

    // CHECK: [[SEND1_TOKEN:%.*]] = "mhlo.send"([[TRUE_REGION_ARG_ELEMENT0]], [[SEND0_TOKEN]])
    "tf.XlaSendToHost"(%arg2) {key = "send_key1"} : (tensor<f32>) -> ()

    // CHECK: [[TRUE_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[TRUE_REGION_ARG_ELEMENT0]], [[SEND1_TOKEN]])
    // CHECK: "mhlo.return"([[TRUE_RETURN_TUPLE]])
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg2: tensor<f32>):
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Tests `mhlo.if` containing TF/XLA communication ops followed by other TF/XLA
// communication ops.

func @if_followed_by_communication_op(%arg0: tensor<i1>, %arg1: tensor<f32>) {
  // CHECK: [[IF_TUPLE:%.*]] = "mhlo.if"
  %0 = "mhlo.if"(%arg0, %arg1, %arg1) ( {
  ^bb0(%arg2: tensor<f32>):
    "tf.XlaSendToHost"(%arg2) {key = "send_key0"} : (tensor<f32>) -> ()
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg2: tensor<f32>):
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: [[IF_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[IF_TUPLE]]) {index = 1

  // CHECK: "mhlo.send"({{.*}}, [[IF_TUPLE_ELEMENT1]])
  "tf.XlaSendToHost"(%arg1) {key = "send_key1"} : (tensor<f32>) -> ()
  return
}

// -----

// Tests `mhlo.while` with cond and body populated with TF/XLA communication
// ops.

// CHECK-LABEL: func @while_cond_body
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func @while_cond_body(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[ARG_TUPLE:%.*]] = "mhlo.tuple"([[ARG0]], [[INIT_TOKEN]])

  // CHECK: [[WHILE_TUPLE:%.*]] = "mhlo.while"([[ARG_TUPLE]])
  %0 = "mhlo.while"(%arg0) ( {
  // CHECK: ^bb0([[COND_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[COND_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[COND_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 1

    // CHECK:      [[COND_SEND_TOKEN:%.*]] = "mhlo.send"([[COND_REGION_ARG_VALUE]], [[COND_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_cond_dtoh_0"}

    // CHECK:      [[COND_RECV_TUPLE:%.*]] = "mhlo.recv"([[COND_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_cond_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_cond", send_key = "send_while_cond", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[COND_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[COND_RECV_TUPLE]]) {index = 1

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[COND_GET_TUPLE_ELEMENT0]], [[COND_GET_TUPLE_ELEMENT0]])
    %2 = "mhlo.compare"(%1, %1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0([[BODY_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[BODY_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[BODY_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 1

    // CHECK:      [[BODY_SEND_TOKEN:%.*]] = "mhlo.send"([[BODY_REGION_ARG_VALUE]], [[BODY_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 3 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_body_dtoh_0"}

    // CHECK:      [[BODY_RECV_TUPLE:%.*]] = "mhlo.recv"([[BODY_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 4 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_body_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_body", send_key = "send_while_body", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[BODY_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[BODY_RECV_TUPLE]]) {index = 1
    // CHECK:      [[BODY_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[BODY_GET_TUPLE_ELEMENT0]], [[BODY_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[BODY_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK: (tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      [[WHILE_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[WHILE_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[WHILE_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` with only the `cond` region populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @while_cond
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func @while_cond(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[ARG_TUPLE:%.*]] = "mhlo.tuple"([[ARG0]], [[INIT_TOKEN]])

  // CHECK: [[WHILE_TUPLE:%.*]] = "mhlo.while"([[ARG_TUPLE]])
  %0 = "mhlo.while"(%arg0) ( {
  // CHECK: ^bb0([[COND_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[COND_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[COND_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 1

    // CHECK:      [[COND_SEND_TOKEN:%.*]] = "mhlo.send"([[COND_REGION_ARG_VALUE]], [[COND_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_cond_dtoh_0"}

    // CHECK:      [[COND_RECV_TUPLE:%.*]] = "mhlo.recv"([[COND_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_cond_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_cond", send_key = "send_while_cond", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[COND_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[COND_RECV_TUPLE]]) {index = 1

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[COND_GET_TUPLE_ELEMENT0]], [[COND_GET_TUPLE_ELEMENT0]])
    %2 = "mhlo.compare"(%1, %1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0([[BODY_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 1
    // CHECK:      [[BODY_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[BODY_GET_TUPLE_ELEMENT0]], [[BODY_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[BODY_RETURN_TUPLE]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  // CHECK: (tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      [[WHILE_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[WHILE_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[WHILE_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` with only the `body` region populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @while_body
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func @while_body(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[ARG_TUPLE:%.*]] = "mhlo.tuple"([[ARG0]], [[INIT_TOKEN]])

  // CHECK: [[WHILE_TUPLE:%.*]] = "mhlo.while"([[ARG_TUPLE]])
  %0 = "mhlo.while"(%arg0) ( {
  // CHECK: ^bb0([[COND_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[COND_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[COND_REGION_ARG]]) {index = 1

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[COND_GET_TUPLE_ELEMENT0]], [[COND_GET_TUPLE_ELEMENT0]])
    %2 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0([[BODY_REGION_ARG:%.*]]: tuple<tensor<f32>, !mhlo.token>):
  ^bb0(%arg1: tensor<f32>):
    // CHECK-DAG:  [[BODY_REGION_ARG_VALUE:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 0
    // CHECK-DAG:  [[BODY_REGION_ARG_TOKEN:%.*]] = "mhlo.get_tuple_element"([[BODY_REGION_ARG]]) {index = 1

    // CHECK:      [[BODY_SEND_TOKEN:%.*]] = "mhlo.send"([[BODY_REGION_ARG_VALUE]], [[BODY_REGION_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 1 : i64, type = 2 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_body_dtoh_0"}

    // CHECK:      [[BODY_RECV_TUPLE:%.*]] = "mhlo.recv"([[BODY_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = {handle = 2 : i64, type = 3 : i64}
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_body_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_body", send_key = "send_while_body", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[BODY_RECV_TUPLE]]) {index = 0
    // CHECK-DAG:  [[BODY_GET_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[BODY_RECV_TUPLE]]) {index = 1
    // CHECK:      [[BODY_RETURN_TUPLE:%.*]] = "mhlo.tuple"([[BODY_GET_TUPLE_ELEMENT0]], [[BODY_GET_TUPLE_ELEMENT1]])
    // CHECK:      "mhlo.return"([[BODY_RETURN_TUPLE]])
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  // CHECK: (tuple<tensor<f32>, !mhlo.token>) -> tuple<tensor<f32>, !mhlo.token>
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      [[WHILE_TUPLE_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[WHILE_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      return [[WHILE_TUPLE_ELEMENT0]]
  return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` containing TF/XLA communication ops followed by other
// TF/XLA communication ops.

func @while_followed_by_communication_op(%arg0: tensor<f32>) {
  // CHECK: [[WHILE_TUPLE:%.*]] = "mhlo.while"
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>):
    "tf.XlaSendToHost"(%arg1) {key = "send_key0"} : (tensor<f32>) -> ()
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK: [[WHILE_TUPLE_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[WHILE_TUPLE]]) {index = 1

  // CHECK: "mhlo.send"({{.*}}, [[WHILE_TUPLE_ELEMENT1]])
  "tf.XlaSendToHost"(%arg0) {key = "send_key1"} : (tensor<f32>) -> ()
  return
}

// -----

// Tests unsupported parent of TF/XLA communication op.

func @unsupported_ancestor(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    // expected-error@+1 {{expects ancestor(s) to be of ['mhlo.if', 'func']}}
    "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> ()
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return
}

// -----

// Tests transitive unsupported parent of TF/XLA communication op.

func @unsupported_ancestor(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    // expected-error@+1 {{expects ancestor(s) to be of ['mhlo.if', 'func']}}
    call @callee() : () -> ()
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return
}

func private @callee() {
  "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> ()
  return
}

// -----

// Tests function with more than one block that is to be rewritten emits an
// error instead.

// expected-error@+1 {{'func' ops with more than one block are not supported}}
func @multi_block_func() {
  br ^bb1
^bb1:
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>
  return
}
