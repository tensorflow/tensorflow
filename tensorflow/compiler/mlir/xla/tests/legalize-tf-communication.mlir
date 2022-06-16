// RUN: xla-opt -split-input-file -verify-diagnostics -xla-legalize-tf-communication %s | FileCheck %s

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
func.func @host_compute(%arg0: tensor<i32>, %arg1: tensor<i64>) -> (tensor<f32>, tensor<f64>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_0"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_ARG1_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 2>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s64", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_1"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i64>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_SINK_TOKEN:%.*]] = "mhlo.after_all"([[SEND_ARG0_TOKEN]], [[SEND_ARG1_TOKEN]])

  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND_SINK_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 3, type = 3>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "host_compute_channel_recv_htod_0"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (!mhlo.token) -> (tensor<f32>, !mhlo.token)

  // CHECK:      [[RECV_RETVAL1_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND_SINK_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 4, type = 3>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f64", _xla_host_transfer_rendezvous = "host_compute_channel_recv_htod_1"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (!mhlo.token) -> (tensor<f64>, !mhlo.token)

  // CHECK:      [[RECV_SINK_TOKEN:%.*]] = "mhlo.after_all"([[RECV_RETVAL0_TUPLE]]#1, [[RECV_RETVAL1_TUPLE]]#1)
  %0:2 = "tf._XlaHostComputeMlir"(%arg0, %arg1) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>, tensor<i64>) -> (tensor<f32>, tensor<f64>)

  // CHECK:      return [[RECV_RETVAL0_TUPLE]]#0, [[RECV_RETVAL1_TUPLE]]#0 : tensor<f32>, tensor<f64>
  func.return %0#0, %0#1 : tensor<f32>, tensor<f64>
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
func.func @host_compute_sharding(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      "mhlo.send"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  // CHECK:      "mhlo.recv"
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\01"
  // CHECK-NOT:      "mhlo.get_tuple_element"
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 1 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// Tests `tf._XlaHostComputeMlir` with no operands simply forwards the input
// token to its generated `mhlo.recv`.

// CHECK-LABEL: func @host_compute_no_operands_one_result
func.func @host_compute_no_operands_one_result() {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK-NOT:  "mhlo.send"
  // CHECK-NOT:  "mhlo.after_all"
  // CHECK:      "mhlo.recv"([[INIT_TOKEN]])
  %0 = "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> tensor<i32>
  func.return
}

// -----

// Tests `tf._XlaHostComputeMlir` with no results simply forwards its token from
// the generated `mhlo.send`.

// CHECK-LABEL: func @host_compute_one_operand_no_results
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func.func @host_compute_one_operand_no_results(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> ()

  // CHECK:      "mhlo.recv"([[SEND_TOKEN]])
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>
  func.return
}

// -----

// Tests `tf._XlaHostComputeMlir` with one operand and one result does not
// create any `mhlo.after_all` ops.

// CHECK-LABEL: func @host_compute_single_operand_result
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func.func @host_compute_single_operand_result(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  // CHECK:      "mhlo.recv"([[SEND_TOKEN]])
  // CHECK-NOT:  "mhlo.after_all"
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>
  func.return
}

// -----

// Test legalization of `tf.XlaSendToHost` expands into a `mhlo.send` op.

// CHECK-LABEL: func @send_to_host
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func.func @send_to_host(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send_key_dtoh_0"}
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> !mhlo.token
  "tf.XlaSendToHost"(%arg0) {key = "send_key"} : (tensor<i32>) -> ()
  func.return
}

// -----

// Test legalization of `tf.XlaRecvFromHost` expands into a `mhlo.recv` op.

// CHECK-LABEL: func @recv_from_host
func.func @recv_from_host() -> tensor<i32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 3>
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv_key_htod_0"}
  // CHECK-SAME: (!mhlo.token) -> (tensor<i32>, !mhlo.token)
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>
  // CHECK:      return [[RECV_TUPLE]]#0 : tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// Test legalization of multiple TF/XLA communication ops are sequenced with
// their generated tokens. Channel Id's are also uniquely assigned.

// CHECK-LABEL: func @multiple_consecutive_ops
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func.func @multiple_consecutive_ops(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND0_ARG0_TOKEN:%.*]] = "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send0_dtoh_0"}

  // CHECK:      [[RECV0_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND0_ARG0_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv0_htod_0"}

  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv0", send_key = "send0", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      [[SEND1_ARG0_TOKEN:%.*]] = "mhlo.send"([[RECV0_RETVAL0_TUPLE]]#0, [[RECV0_RETVAL0_TUPLE]]#1)
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 3, type = 2>
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send1_dtoh_0"}

  // CHECK:      [[RECV1_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND1_ARG0_TOKEN]])
  // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 4, type = 3>
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv1_htod_0"}

  %1 = "tf._XlaHostComputeMlir"(%0) {recv_key = "recv1", send_key = "send1", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV1_RETVAL0_TUPLE]]#0 : tensor<i32>
  func.return %1 : tensor<i32>
}

// -----

// Test private function with TF/XLA communication op being called by another
// function gets rewritten with an extra token argument and an extra token
// result, and the caller passes in a token. The top level function not called
// (or public) will be updated to create a token.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL:%.*]]:2 = call @callee([[MAIN_ARG0]], [[MAIN_TOKEN]])
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> (tensor<i32>, !mhlo.token)
  %0 = func.call @callee(%arg0) : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[CALL]]#0
  func.return %0 : tensor<i32>
}

// CHECK: func private @callee([[CALLEE_ARG0:%.*]]: tensor<i32>, [[CALLEE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
func.func private @callee(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_ARG0]], [[CALLEE_ARG1]])
  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND_ARG0_TOKEN]])
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV_RETVAL0_TUPLE]]#0, [[RECV_RETVAL0_TUPLE]]#1
  func.return %0 : tensor<i32>
}

// -----

// Test public function with TF/XLA communication op being called by another
// function. The original public function will be modified to create a token,
// while the function is cloned and rewritten with an extra token argument and
// an extra token result. All callers to the original function are updated to
// point to the cloned function and the function the caller is in is updated to
// pass a token or create a token.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL:%.*]]:2 = call [[CALLEE_CLONE:@.*]]([[MAIN_ARG0]], [[MAIN_TOKEN]])
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> (tensor<i32>, !mhlo.token)
  %0 = func.call @callee(%arg0) : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[CALL]]#0 : tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK: func @callee([[CALLEE_ARG0:%.*]]: tensor<i32>) -> tensor<i32>
func.func @callee(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK:      [[CALLEE_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_ARG0]], [[CALLEE_TOKEN]])
  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[SEND_ARG0_TOKEN]])
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV_RETVAL0_TUPLE]]#0
  func.return %0 : tensor<i32>
}

// CHECK: func private [[CALLEE_CLONE]]([[CALLEE_CLONE_ARG0:%.*]]: tensor<i32>, [[CALLEE_CLONE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
// CHECK-NOT:  "mhlo.create_token"

// CHECK:      [[CLONE_SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_CLONE_ARG0]], [[CALLEE_CLONE_ARG1]])
// CHECK:      [[CLONE_RECV_RETVAL0_TUPLE:%.*]]:2 = "mhlo.recv"([[CLONE_SEND_ARG0_TOKEN]])

// CHECK:      return [[CLONE_RECV_RETVAL0_TUPLE]]#0, [[CLONE_RECV_RETVAL0_TUPLE]]#1

// -----

// Tests generated tokens are passed into a function call that also has TF/XLA
// communication ops.

// CHECK: func @main([[MAIN_ARG0:%.*]]: tensor<i32>)
func.func @main(%arg0: tensor<i32>) {
  // CHECK:      [[MAIN_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[MAIN_SEND0_TOKEN:%.*]] = "mhlo.send"([[MAIN_ARG0]], [[MAIN_TOKEN]])
  "tf.XlaSendToHost"(%arg0) {key = "send0"} : (tensor<i32>) -> ()

  // CHECK:      [[CALL_TOKEN:%.*]] = call @callee([[MAIN_SEND0_TOKEN]])
  // CHECK-SAME: (!mhlo.token) -> !mhlo.token
  func.call @callee() : () -> ()

  // CHECK:      [[MAIN_SEND2_TOKEN:%.*]] = "mhlo.send"([[MAIN_ARG0]], [[CALL_TOKEN]])
  "tf.XlaSendToHost"(%arg0) {key = "send2"} : (tensor<i32>) -> ()
  func.return
}

// CHECK: func private @callee([[CALLEE_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func.func private @callee() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[ZERO:%.*]] = mhlo.constant dense<0>
  %0 = mhlo.constant dense<0> : tensor<i32>

  // CHECK:      [[CALLEE_SEND_TOKEN:%.*]] = "mhlo.send"([[ZERO]], [[CALLEE_ARG0]])
  "tf.XlaSendToHost"(%0) {key = "send1"} : (tensor<i32>) -> ()

  // CHECK:      return [[CALLEE_SEND_TOKEN]]
  func.return
}

// -----

// Test only the top level function generates a token.

// CHECK: func private @callee0()
func.func private @callee0() {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      call @callee1([[INIT_TOKEN]])
  func.call @callee1() : () -> ()
  func.return
}

// CHECK: func private @callee1([[CALLEE1_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func.func private @callee1() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[CALL_2:%.*]] = call @callee2([[CALLEE1_ARG0]])
  func.call @callee2() : () -> ()

  // CHECK:      return [[CALL_2]]
  func.return
}

// CHECK: func private @callee2([[CALLEE2_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func.func private @callee2() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[CALLEE2_ARG0]])
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>

  // CHECK:      return [[RECV_TUPLE]]#1
  func.return
}

// -----

// Test cloned function rewrite also checks transitive function calls to
// TF/XLA communication ops.

// CHECK: func @callee3()
func.func @callee3() {
  // CHECK:      [[CALLEE3_INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      call @callee4{{.+}}([[CALLEE3_INIT_TOKEN]])
  func.call @callee4() : () -> ()
  func.return
}

// CHECK: func @callee4()
func.func @callee4() {
  // CHECK:      [[CALLEE4_INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[CALL_5:%.*]] = call @callee5([[CALLEE4_INIT_TOKEN]])
  func.call @callee5() : () -> ()

  // CHECK:      return
  func.return
}

// CHECK: func private @callee5([[CALLEE5_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func.func private @callee5() {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[CALLEE5_ARG0]])
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>

  // CHECK:      return [[RECV_TUPLE]]#1
  func.return
}

// CHECK: func private @callee4{{.+}}([[CALLEE4_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
// CHECK-NOT:  "mhlo.create_token"
// CHECK:      [[CALL_5:%.*]] = call @callee5([[CALLEE4_ARG0]])
// CHECK:      return [[CALL_5]]

// -----

// Tests `mhlo.if` with branches populated with TF/XLA communication ops.

// CHECK-LABEL: func @if_both_branches
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func.func @if_both_branches(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK: [[IF:%.*]]:2 = "mhlo.if"([[ARG0]])
  %0 = "mhlo.if"(%arg0) ({
    // CHECK:      [[TRUE_SEND_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_true_dtoh_0"}

    // CHECK:      [[TRUE_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[TRUE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_true_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_if_true", send_key = "send_if_true", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[TRUE_RECV_TUPLE]]#0, [[TRUE_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
    // CHECK:      [[FALSE_SEND_TOKEN:%.*]] = "mhlo.send"([[ARG2]], [[INIT_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 3, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_false_dtoh_0"}

    // CHECK:      [[FALSE_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[FALSE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 4, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_false_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg2) {recv_key = "recv_if_false", send_key = "send_if_false", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[FALSE_RECV_TUPLE]]#0, [[FALSE_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>) -> (tensor<f32>, !mhlo.token)
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:      return [[IF]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with only the `true` branch populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @if_true_branch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func.func @if_true_branch(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK: [[IF:%.*]]:2 = "mhlo.if"([[ARG0]])
  %0 = "mhlo.if"(%arg0) ({
    // CHECK:      [[TRUE_SEND_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_true_dtoh_0"}

    // CHECK:      [[TRUE_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[TRUE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_true_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_if_true", send_key = "send_if_true", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[TRUE_RECV_TUPLE]]#0, [[TRUE_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  },  {
    // CHECK:      "mhlo.return"([[ARG2]], [[INIT_TOKEN]])
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>) -> (tensor<f32>, !mhlo.token)
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:      return [[IF]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with only the `false` branch populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @if_false_branch
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func.func @if_false_branch(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK: [[IF:%.*]]:2 = "mhlo.if"([[ARG0]])
  %0 = "mhlo.if"(%arg0) ({
    // CHECK:      "mhlo.return"([[ARG1]], [[INIT_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    // CHECK:      [[FALSE_SEND_TOKEN:%.*]] = "mhlo.send"([[ARG2]], [[INIT_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_if_false_dtoh_0"}

    // CHECK:      [[FALSE_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[FALSE_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_if_false_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg2) {recv_key = "recv_if_false", send_key = "send_if_false", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[FALSE_RECV_TUPLE]]#0, [[FALSE_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  // CHECK: (tensor<i1>) -> (tensor<f32>, !mhlo.token)
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK:      return [[IF]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` with tuple arg from a `mhlo.tuple` only used by `mhlo.if` is
// replaced.

// CHECK-LABEL: func @if_replace_tuple_arg
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<f32>)
func.func @if_replace_tuple_arg(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK-NOT:  "mhlo.tuple"([[ARG1]], [[ARG2]])
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK: [[IF:%.*]] = "mhlo.if"([[ARG0]])
  %1 = "mhlo.if"(%arg0) ({
    "tf.XlaSendToHost"(%arg1) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// -----

// Tests `mhlo.if` with tuple arg not from a `mhlo.tuple` is unpacked.

// CHECK-LABEL: func @if_unpack_tuple_arg
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tuple<tensor<f32>, tensor<f32>>)
func.func @if_unpack_tuple_arg(%arg0: tensor<i1>, %arg1: tuple<tensor<f32>, tensor<f32>>) -> tensor<f32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK-DAG:  [[IF_ARG_ELEMENT0:%.*]] = "mhlo.get_tuple_element"([[ARG1]]) {index = 0
  // CHECK-DAG:  [[IF_ARG_ELEMENT1:%.*]] = "mhlo.get_tuple_element"([[ARG1]]) {index = 1
  %0 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  %1 = "mhlo.get_tuple_element"(%arg1) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>

  // CHECK:      [[IF:%.*]] = "mhlo.if"([[ARG0]])
  %2 = "mhlo.if"(%arg0) ({
    "tf.XlaSendToHost"(%0) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%0) : (tensor<f32>) -> ()
  },  {
    "mhlo.return"(%0) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// Tests `mhlo.if` tuple result is extended with a `mhlo.token`.

// CHECK-LABEL: func @if_extend_tuple_result
func.func @if_extend_tuple_result(%arg0: tensor<i1>, %arg1: tuple<tensor<f32>, tensor<f32>>) -> tuple<tensor<f32>, tensor<f32>> {
  %0 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  %1 = "mhlo.get_tuple_element"(%arg1) {index = 1 : i32} : (tuple<tensor<f32>, tensor<f32>>) -> tensor<f32>
  // CHECK:      [[IF:%.*]]:3 = "mhlo.if"
  %2:2 = "mhlo.if"(%arg0) ({
    "tf.XlaSendToHost"(%0) {key = "send_key"} : (tensor<f32>) -> ()
    "mhlo.return"(%0, %1) : (tensor<f32>, tensor<f32>) -> ()
  },  {
    "mhlo.return"(%0, %1) : (tensor<f32>, tensor<f32>) -> ()
  // CHECK:      (tensor<i1>) -> (tensor<f32>, tensor<f32>, !mhlo.token)
  }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)

  // CHECK:      [[IF_SUBTUPLE_RESULT:%.*]] = "mhlo.tuple"([[IF]]#0, [[IF]]#1)
  %3 = "mhlo.tuple"(%2#0, %2#1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
  // CHECK:      return [[IF_SUBTUPLE_RESULT]]
  func.return %3 : tuple<tensor<f32>, tensor<f32>>
}

// -----

// Tests nested `mhlo.if` containing TF/XLA communication ops.

// CHECK-LABEL: func @if_nested
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>)
func.func @if_nested(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      [[IF:%.*]]:2 = "mhlo.if"([[ARG0]])
  %0 = "mhlo.if"(%arg0) ({
    %1 = mhlo.constant dense<false> : tensor<i1>

    // CHECK:      [[INNER_IF:%.*]]:2 = "mhlo.if"({{%.*}})
    %2 = "mhlo.if"(%1) ({

      // CHECK:      [[SEND_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
      "tf.XlaSendToHost"(%arg1) {key = "send_key"} : (tensor<f32>) -> ()

      // CHECK:      "mhlo.return"([[ARG1]], [[SEND_TOKEN]])
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()

    // CHECK-NEXT: }, {
    },  {

      // CHECK:      "mhlo.return"([[ARG1]], [[INIT_TOKEN]])
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    // CHECK-NEXT: (tensor<i1>) -> (tensor<f32>, !mhlo.token)
    }) : (tensor<i1>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[ARG1]], [[INNER_IF]]#1)
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()

  // CHECK-NEXT: },  {
  },  {

    // CHECK:      "mhlo.return"([[ARG1]], [[INIT_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  // CHECK-NEXT: (tensor<i1>) -> (tensor<f32>, !mhlo.token)
  }) : (tensor<i1>) -> tensor<f32>
  // CHECK-NEXT: return [[IF]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.if` containing a function call to TF/XLA communication ops.

// CHECK-LABEL: func @if_function_call
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>)
func.func @if_function_call(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: "mhlo.if"
  %0 = "mhlo.if"(%arg0) ({
    // CHECK:      [[CALL_TOKEN:%.*]] = func.call @callee([[ARG1]], [[INIT_TOKEN]])
    func.call @callee(%arg1) : (tensor<f32>) -> ()

    // CHECK:      "mhlo.return"([[ARG1]], [[CALL_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func private @callee
// CHECK-SAME:  ([[CALLEE_ARG0:%.*]]: tensor<f32>, [[CALLEE_ARG1:%.*]]: !mhlo.token) -> !mhlo.token
func.func private @callee(%arg0: tensor<f32>) {
  // CHECK: [[SEND_TOKEN:%.*]] = "mhlo.send"
  "tf.XlaSendToHost"(%arg0) {key = "send_key"} : (tensor<f32>) -> ()

  // CHECK: return [[SEND_TOKEN]]
  func.return
}

// -----

// Tests `mhlo.if` containing multiple TF/XLA communication ops.

// CHECK-LABEL: func @if_region_multiple_ops
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>)
func.func @if_region_multiple_ops(%arg0: tensor<i1>, %arg1: tensor<f32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: "mhlo.if"
  %0 = "mhlo.if"(%arg0) ({
    // CHECK: [[SEND0_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
    "tf.XlaSendToHost"(%arg1) {key = "send_key0"} : (tensor<f32>) -> ()

    // CHECK: [[SEND1_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[SEND0_TOKEN]])
    "tf.XlaSendToHost"(%arg1) {key = "send_key1"} : (tensor<f32>) -> ()

    // CHECK: "mhlo.return"([[ARG1]], [[SEND1_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  },  {
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return
}

// -----

// Tests `mhlo.if` containing TF/XLA communication ops followed by other TF/XLA
// communication ops.

// CHECK-LABEL: func @if_followed_by_communication_op
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i1>, [[ARG1:%.*]]: tensor<f32>)
func.func @if_followed_by_communication_op(%arg0: tensor<i1>, %arg1: tensor<f32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK-NEXT: [[IF:%.*]]:2 = "mhlo.if"
  %0 = "mhlo.if"(%arg0) ({
    // CHECK-NEXT: [[SEND0_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
    "tf.XlaSendToHost"(%arg1) {key = "send_key0"} : (tensor<f32>) -> ()
    // CHECK-NEXT:      "mhlo.return"([[ARG1]], [[SEND0_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    // CHECK-NEXT: },  {
  },  {
    // CHECK-NEXT:      "mhlo.return"([[ARG1]], [[INIT_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
    // CHECK-NEXT:      })
  }) : (tensor<i1>) -> tensor<f32>

  // CHECK-NEXT: "mhlo.send"({{.*}}, [[IF]]#1)
  "tf.XlaSendToHost"(%arg1) {key = "send_key1"} : (tensor<f32>) -> ()
  func.return
}

// -----

// Tests `mhlo.while` with cond and body populated with TF/XLA communication
// ops.

// CHECK-LABEL: func @while_cond_body
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func.func @while_cond_body(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK: [[WHILE:%.*]]:2 = mhlo.while([[ITER_ARG_VALUE:.*]] = [[ARG0]], [[ITER_ARG_TOKEN:.*]] = [[INIT_TOKEN]])
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<f32>):
    // CHECK:      [[COND_SEND_TOKEN:%.*]] = "mhlo.send"([[ITER_ARG_VALUE]], [[ITER_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_cond_dtoh_0"}

    // CHECK:      [[COND_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[COND_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_cond_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_cond", send_key = "send_while_cond", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[COND_RECV_TUPLE]]#0, [[COND_RECV_TUPLE]]#0)
    %2 = "mhlo.compare"(%1, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):
    // CHECK:      [[BODY_SEND_TOKEN:%.*]] = "mhlo.send"([[ITER_ARG_VALUE]], [[ITER_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 3, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_body_dtoh_0"}

    // CHECK:      [[BODY_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[BODY_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 4, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_body_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_body", send_key = "send_while_body", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[BODY_RECV_TUPLE]]#0, [[BODY_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      return [[WHILE]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` with only the `cond` region populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @while_cond
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func.func @while_cond(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[WHILE:%.*]]:2 = mhlo.while([[ITER_ARG_VALUE:.*]] = [[ARG0]], [[ITER_ARG_TOKEN:.*]] = [[INIT_TOKEN]])
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<f32>):

    // CHECK:      [[COND_SEND_TOKEN:%.*]] = "mhlo.send"([[ITER_ARG_VALUE]], [[ITER_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_cond_dtoh_0"}

    // CHECK:      [[COND_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[COND_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_cond_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_cond", send_key = "send_while_cond", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[COND_RECV_TUPLE]]#0, [[COND_RECV_TUPLE]]#0)
    %2 = "mhlo.compare"(%1, %1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):
    // CHECK:  "mhlo.return"([[ITER_ARG_VALUE]], [[ITER_ARG_TOKEN]])
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      return [[WHILE]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` with only the `body` region populated with TF/XLA
// communication ops.

// CHECK-LABEL: func @while_body
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<f32>)
func.func @while_body(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT_TOKEN:%.*]] = "mhlo.create_token"
  // CHECK: [[WHILE:%.*]]:2 = mhlo.while([[ITER_ARG_VALUE:.*]] = [[ARG0]], [[ITER_ARG_TOKEN:.*]] = [[INIT_TOKEN]])
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<f32>):

    // CHECK:      [[COND_COMPARE:%.*]] = "mhlo.compare"([[ITER_ARG_VALUE]], [[ITER_ARG_VALUE]])
    %2 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>

    // CHECK:      "mhlo.return"([[COND_COMPARE]])
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):

    // CHECK:      [[BODY_SEND_TOKEN:%.*]] = "mhlo.send"([[ITER_ARG_VALUE]], [[ITER_ARG_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 1, type = 2>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "send_while_body_dtoh_0"}

    // CHECK:      [[BODY_RECV_TUPLE:%.*]]:2 = "mhlo.recv"([[BODY_SEND_TOKEN]])
    // CHECK-SAME: channel_handle = #mhlo.channel_handle<handle = 2, type = 3>
    // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_original_type = "f32", _xla_host_transfer_rendezvous = "recv_while_body_htod_0"}
    %1 = "tf._XlaHostComputeMlir"(%arg1) {recv_key = "recv_while_body", send_key = "send_while_body", tpu_core = 0 : i64, host_mlir_module = ""} : (tensor<f32>) -> tensor<f32>

    // CHECK:      "mhlo.return"([[BODY_RECV_TUPLE]]#0, [[BODY_RECV_TUPLE]]#1)
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK:      return [[WHILE]]#0
  func.return %0 : tensor<f32>
}

// -----

// Tests `mhlo.while` containing TF/XLA communication ops followed by other
// TF/XLA communication ops.

// CHECK-LABEL: func @while_followed_by_communication_op
func.func @while_followed_by_communication_op(%arg0: tensor<f32>) {
  // CHECK: [[WHILE:%.*]]:2 = mhlo.while
  %0 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<f32>):
    "tf.XlaSendToHost"(%arg1) {key = "send_key0"} : (tensor<f32>) -> ()
    %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<f32>):
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>

  // CHECK: "mhlo.send"({{.*}}, [[WHILE]]#1)
  "tf.XlaSendToHost"(%arg0) {key = "send_key1"} : (tensor<f32>) -> ()
  func.return
}

// -----

// Tests unsupported parent of TF/XLA communication op.

func.func @unsupported_ancestor(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    // expected-error@+1 {{expects ancestor(s) to be of ['mhlo.if', 'func.func']}}
    "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> ()
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  func.return
}

// -----

// Tests transitive unsupported parent of TF/XLA communication op.

func.func @unsupported_ancestor(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    // expected-error@+1 {{expects ancestor(s) to be of ['mhlo.if', 'func.func']}}
    func.call @callee() : () -> ()
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  func.return
}

func.func private @callee() {
  "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64, host_mlir_module = ""} : () -> ()
  func.return
}

// -----

// Tests function with more than one block that is to be rewritten emits an
// error instead.

// expected-error@+1 {{'func.func' ops with more than one block are not supported}}
func.func @multi_block_func() {
  cf.br ^bb1
^bb1:
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf_type.shape<>} : () -> tensor<i32>
  func.return
}
