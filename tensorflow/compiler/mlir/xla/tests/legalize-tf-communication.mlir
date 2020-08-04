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
  // CHECK-SAME: channel_id = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_0"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i32>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_ARG1_TOKEN:%.*]] = "mhlo.send"([[ARG1]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 2 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s64", _xla_host_transfer_rendezvous = "host_compute_channel_send_dtoh_1"}
  // CHECK-SAME: mhlo.sharding = "\08\01\1A\01\01\22\01\00"
  // CHECK-SAME: (tensor<i64>, !mhlo.token) -> !mhlo.token

  // CHECK:      [[SEND_SINK_TOKEN:%.*]] = "mhlo.after_all"([[SEND_ARG0_TOKEN]], [[SEND_ARG1_TOKEN]])

  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND_SINK_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 3 : i64, type = 3 : i64}
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
  // CHECK-SAME: channel_id = {handle = 4 : i64, type = 3 : i64}
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
  %0:2 = "tf._XlaHostComputeMlir"(%arg0, %arg1) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64} : (tensor<i32>, tensor<i64>) -> (tensor<f32>, tensor<f64>)

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
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 1 : i64} : (tensor<i32>) -> tensor<i32>
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
  %0 = "tf._XlaHostComputeMlir"() {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64} : () -> tensor<i32>
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
  "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64} : (tensor<i32>) -> ()

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
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "host_compute_channel_recv", send_key = "host_compute_channel_send", tpu_core = 0 : i64} : (tensor<i32>) -> tensor<i32>
  return
}

// -----

// Test legalization of `tf.XlaSendToHost` expands into a `mhlo.send` op.

// CHECK-LABEL: func @send_to_host
// CHECK-SAME:  ([[ARG0:%.*]]: tensor<i32>)
func @send_to_host(%arg0: tensor<i32>) {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      "mhlo.send"([[ARG0]], [[INIT_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send_key"}
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
  // CHECK-SAME: channel_id = {handle = 1 : i64, type = 3 : i64}
  // CHECK-SAME: is_host_transfer = true
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv_key"}
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
  // CHECK-SAME: channel_id = {handle = 1 : i64, type = 2 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send0_dtoh_0"}

  // CHECK:      [[RECV0_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND0_ARG0_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 2 : i64, type = 3 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv0_htod_0"}

  // CHECK:      [[RECV0_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV0_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0

  // CHECK:      [[RECV0_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV0_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv0", send_key = "send0", tpu_core = 0 : i64} : (tensor<i32>) -> tensor<i32>

  // CHECK:      [[SEND1_ARG0_TOKEN:%.*]] = "mhlo.send"([[RECV0_RETVAL0_VAL]], [[RECV0_RETVAL0_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 3 : i64, type = 2 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "send1_dtoh_0"}

  // CHECK:      [[RECV1_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND1_ARG0_TOKEN]])
  // CHECK-SAME: channel_id = {handle = 4 : i64, type = 3 : i64}
  // CHECK-SAME: mhlo.frontend_attributes = {_xla_host_transfer_original_type = "s32", _xla_host_transfer_rendezvous = "recv1_htod_0"}

  // CHECK:      [[RECV1_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV1_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0

  // CHECK:      [[RECV1_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV1_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %1 = "tf._XlaHostComputeMlir"(%0) {recv_key = "recv1", send_key = "send1", tpu_core = 0 : i64} : (tensor<i32>) -> tensor<i32>

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

// CHECK: func @callee([[CALLEE_ARG0:%.*]]: tensor<i32>, [[CALLEE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
func @callee(%arg0: tensor<i32>) -> tensor<i32> attributes {sym_visibility = "private"} {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[SEND_ARG0_TOKEN:%.*]] = "mhlo.send"([[CALLEE_ARG0]], [[CALLEE_ARG1]])
  // CHECK:      [[RECV_RETVAL0_TUPLE:%.*]] = "mhlo.recv"([[SEND_ARG0_TOKEN]])
  // CHECK:      [[RECV_RETVAL0_VAL:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 0
  // CHECK:      [[RECV_RETVAL0_TOKEN:%.*]] = "mhlo.get_tuple_element"([[RECV_RETVAL0_TUPLE]])
  // CHECK-SAME: index = 1
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64} : (tensor<i32>) -> tensor<i32>

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
  %0 = "tf._XlaHostComputeMlir"(%arg0) {recv_key = "recv", send_key = "send", tpu_core = 0 : i64} : (tensor<i32>) -> tensor<i32>

  // CHECK:      return [[RECV_RETVAL0_VAL]]
  return %0 : tensor<i32>
}

// CHECK: func [[CALLEE_CLONE]]([[CALLEE_CLONE_ARG0:%.*]]: tensor<i32>, [[CALLEE_CLONE_ARG1:%.*]]: !mhlo.token) -> (tensor<i32>, !mhlo.token)
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

// CHECK: func @callee([[CALLEE_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func @callee() attributes {sym_visibility = "private"} {
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

// CHECK: func @callee0()
func @callee0() attributes {sym_visibility = "private"} {
  // CHECK:      [[INIT_TOKEN:%.*]] = "mhlo.create_token"

  // CHECK:      call @callee1([[INIT_TOKEN]])
  call @callee1() : () -> ()
  return
}

// CHECK: func @callee1([[CALLEE1_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func @callee1() attributes {sym_visibility = "private"} {
  // CHECK-NOT:  "mhlo.create_token"

  // CHECK:      [[CALL_2:%.*]] = call @callee2([[CALLEE1_ARG0]])
  call @callee2() : () -> ()

  // CHECK:      return [[CALL_2]]
  return
}

// CHECK: func @callee2([[CALLEE2_ARG0:%.*]]: !mhlo.token) -> !mhlo.token
func @callee2() attributes {sym_visibility = "private"} {
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

// Tests function with more than one block that is to be rewritten emits an
// error instead.

// expected-error@+1 {{'func' ops with more than one block are not supported}}
func @multi_block_func() {
  br ^bb1
^bb1:
  %0 = "tf.XlaRecvFromHost"() {key = "recv_key", shape = #tf.shape<>} : () -> tensor<i32>
  return
}
