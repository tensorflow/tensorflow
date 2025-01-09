// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// CHECK:  HloModule
func.func @all_gather_0(%arg1: tensor<128x32xf32>) -> tensor<128x128xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.all_gather"(%arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @all_gather_0, execution_thread = "main"} : (tensor<128x32xf32>) -> !mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: %[[OUTPUT:.*]] = f32[128,128] all-gather-start(f32[128,32] %[[INPUT]])
// CHECK-SAME: channel_id=1
// CHECK-SAME{LITERAL}: replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: dimensions={1}
// CHECK-SAME: use_global_device_ids=true
// CHECK: ROOT {{.*}} f32[128,128] all-gather-done(f32[128,128] %[[OUTPUT]]

// -----

// CHECK:  HloModule
func.func @all_reduce_0(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @all_reduce_0, execution_thread = "main"} : (tensor<10xf32>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[10] parameter(0)
// CHECK: %[[OUTPUT:.*]] = f32[10] all-reduce-start(f32[10] %[[INPUT]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: use_global_device_ids=true
// CHECK: ROOT {{.*}} f32[10] all-reduce-done(f32[10] %[[OUTPUT]]

// -----

// expected-error@-3 {{'mhlo.async_start' op can't be translated to XLA HLO}}
func.func @all_reduce_0(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>) -> (tensor<10xf32>, tensor<1xf32>) attributes {execution_thread = "main"} {
  %0:2 = "mhlo.all_reduce"(%arg0, %arg1) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>, tensor<1xf32>) -> (tensor<10xf32>, tensor<1xf32>)
  func.return %0#0, %0#1 : tensor<10xf32>, tensor<1xf32>
}

func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>) -> (tensor<10xf32>, tensor<1xf32>) {
  %0 = "mhlo.async_start"(%arg0, %arg1) {called_computation = @all_reduce_0, execution_thread = "main"} : (tensor<10xf32>, tensor<1xf32>) -> !mhlo.async_bundle<tuple<tensor<10xf32>,tensor<1xf32>>, tuple<tensor<10xf32>,tensor<1xf32>>>
  %1:2 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tuple<tensor<10xf32>,tensor<1xf32>>, tuple<tensor<10xf32>,tensor<1xf32>>>) -> (tensor<10xf32>, tensor<1xf32>)
  return %1#0, %1#1 : tensor<10xf32>, tensor<1xf32>
}

// -----

// expected-error@-3 {{'mhlo.async_start' op can't be translated to XLA HLO}}
func.func @all_gather_0(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> (tensor<8x2xf32>, tensor<8x4xf32>) attributes {execution_thread = "main"} {
  %0:2 = "mhlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<8x2xf32>, tensor<8x4xf32>) -> (tensor<8x2xf32>, tensor<8x4xf32>)
  func.return %0#0, %0#1 : tensor<8x2xf32>, tensor<8x4xf32>
}

func.func @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> (tensor<8x2xf32>, tensor<8x4xf32>) {
  %0 = "mhlo.async_start"(%arg0, %arg1) {called_computation = @all_gather_0, execution_thread = "main"} : (tensor<8x2xf32>, tensor<8x4xf32>) -> !mhlo.async_bundle<tuple<tensor<8x2xf32>,tensor<8x4xf32>>, tuple<tensor<8x2xf32>,tensor<8x4xf32>>>
  %1:2 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tuple<tensor<8x2xf32>,tensor<8x4xf32>>, tuple<tensor<8x2xf32>,tensor<8x4xf32>>>) -> (tensor<8x2xf32>, tensor<8x4xf32>)
  return %1#0, %1#1 : tensor<8x2xf32>, tensor<8x4xf32>
}

// -----

// CHECK:  HloModule
func.func @collective_permute_0(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @collective_permute_0, execution_thread = "main"} : (tensor<128x32xf32>) -> !mhlo.async_bundle<tensor<128x32xf32>, tensor<128x32xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<128x32xf32>, tensor<128x32xf32>>) -> tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: %[[OUTPUT:.*]] = f32[128,32] collective-permute-start(f32[128,32] %[[INPUT]])
// CHECK-SAME:  channel_id=1
// CHECK-SAME{LITERAL}:  source_target_pairs={{0,1},{1,2},{2,3}}
// CHECK: ROOT {{.*}} f32[128,32] collective-permute-done(f32[128,32] %[[OUTPUT]]

// -----

// CHECK:  HloModule
func.func @copy_0(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.copy"(%arg0) {cross_program_prefetch_index = 0 : i32} : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @copy_0, execution_thread = "main"} : (tensor<128x32xf32>) -> !mhlo.async_bundle<tensor<128x32xf32>, tensor<128x32xf32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tensor<128x32xf32>, tensor<128x32xf32>>) -> tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: %[[OUTPUT:.*]] = (f32[128,32], f32[128,32], u32[]) copy-start(f32[128,32] %[[INPUT]])
// CHECK-SAME:  cross_program_prefetch_index=0
// CHECK: ROOT {{.*}} f32[128,32] copy-done((f32[128,32], f32[128,32], u32[]) %[[OUTPUT]]

// -----

// CHECK:  HloModule

func.func @recv_0(%token: !mhlo.token) -> (!mhlo.token) attributes {execution_thread = "main"} {
  %0 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (!mhlo.token) -> (!mhlo.token)
  func.return %0 : !mhlo.token
}

func.func @main(%token: !mhlo.token) -> (!mhlo.token) {
  %0 = "mhlo.async_start"(%token) {called_computation = @recv_0, execution_thread = "main"} : (!mhlo.token) -> !mhlo.async_bundle<!mhlo.token, !mhlo.token, tensor<i32>>
  %2 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<!mhlo.token, !mhlo.token, tensor<i32>>) -> (!mhlo.token)
  return %2 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = ((), u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5
// CHECK:  ((), token[]) recv-done(((), u32[], token[]) [[RECV]]), channel_id=5

// -----

// CHECK:  HloModule
func.func @recv_0(%token: !mhlo.token) -> (tensor<3x4xi32>, !mhlo.token) attributes {execution_thread = "main"} {
  %0:2 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >,
    is_host_transfer = true
  } : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
  func.return %0#0, %0#1 : tensor<3x4xi32>, !mhlo.token
}

func.func @main(%token: !mhlo.token) -> (tensor<3x4xi32>, !mhlo.token) {
  %0 = "mhlo.async_start"(%token) {called_computation = @recv_0, execution_thread = "main", mhlo.sharding = "{{maximal device=0}, {maximal device=0}, {maximal device=0}}"} : (!mhlo.token) -> !mhlo.async_bundle<!mhlo.token, tuple<tensor<3x4xi32>, !mhlo.token>, tensor<i32>>
  %1, %2 = "mhlo.async_done"(%0) {mhlo.sharding = "{{maximal device=0}, {maximal device=0}}"} : (!mhlo.async_bundle<!mhlo.token, tuple<tensor<3x4xi32>, !mhlo.token>, tensor<i32>>) -> (tensor<3x4xi32>, !mhlo.token)
  return %1, %2 : tensor<3x4xi32>, !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = (s32[3,4], u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5, is_host_transfer
// CHECK-SAME:  sharding={
// CHECK-SAME:    {maximal device=0}, {maximal device=0}, {maximal device=0}
// CHECK-SAME:  }
// CHECK:  [[RECV_DONE:%.*]] = (s32[3,4], token[]) recv-done((s32[3,4], u32[], token[]) [[RECV]]), channel_id=5, is_host_transfer
// CHECK-SAME:  sharding={
// CHECK-SAME:    {maximal device=0}, {maximal device=0}
// CHECK-SAME:  }
// CHECK:  [[TUPLE0:%.*]] = s32[3,4] get-tuple-element((s32[3,4], token[]) [[RECV_DONE]]), index=0, sharding={maximal device=0}
// CHECK:  [[TUPLE1:%.*]] = token[] get-tuple-element((s32[3,4], token[]) [[RECV_DONE]]), index=1, sharding={maximal device=0}
// CHECK:  ROOT {{%.*}} = (s32[3,4], token[]) tuple(s32[3,4] [[TUPLE0]], token[] [[TUPLE1]])

// -----

// CHECK:  HloModule
func.func @send_0(%arg: tensor<3x4xi32>, %token: !mhlo.token) -> !mhlo.token attributes {execution_thread = "main"} {
  %0 = "mhlo.send"(%arg, %token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2  // Device to host channel
    >,
    is_host_transfer = true
  } : (tensor<3x4xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

func.func @main(%arg: tensor<3x4xi32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.async_start"(%arg, %token) {called_computation = @send_0, execution_thread = "main"} : (tensor<3x4xi32>, !mhlo.token) -> !mhlo.async_bundle<tuple<tensor<3x4xi32>, !mhlo.token>, !mhlo.token, tensor<i32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<tuple<tensor<3x4xi32>, !mhlo.token>, !mhlo.token, tensor<i32>>) -> !mhlo.token
  return %1 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  [[SEND:%.*]] = (s32[3,4], u32[], token[]) send(s32[3,4] [[ARG]], token[] [[TOKEN]]), channel_id=5, is_host_transfer
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done((s32[3,4], u32[], token[]) [[SEND]]), channel_id=5

// -----

// CHECK:  HloModule
func.func @send_0(%token: !mhlo.token) -> !mhlo.token attributes {execution_thread = "main"} {
  %0 = "mhlo.send"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >
  } : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

func.func @main(%token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.async_start"(%token) {called_computation = @send_0, execution_thread = "main"} : (!mhlo.token) -> !mhlo.async_bundle<!mhlo.token, !mhlo.token, tensor<i32>>
  %1 = "mhlo.async_done"(%0) : (!mhlo.async_bundle<!mhlo.token, !mhlo.token, tensor<i32>>) -> !mhlo.token
  return %1 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[SEND:%.*]] = ((), u32[], token[]) send(() [[UNIT:%.*]], token[] [[TOKEN]]), channel_id=5
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done(((), u32[], token[]) [[SEND]]), channel_id=5

// -----

// CHECK: HloModule
// CHECK: [[CALLED_COMPUTATION:%AsyncOp.*]] ([[ARG:.*]]: f32[10]) -> f32[20] {
func.func @AsyncOp(%arg0: tensor<10xf32>) -> tensor<20xf32>
  attributes {execution_thread = "thread"} {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<10xf32>) -> tensor<20xf32>
  return %0 : tensor<20xf32>
}

// CHECK: ENTRY
func.func @main(%arg0: tensor<10xf32>) -> tensor<20xf32> {
  // CHECK: %[[ARG0:.*]] = f32[10] parameter(0)
  // CHECK: %[[START:.*]] = ((f32[10]), f32[20], s32[]) async-start(f32[10] %[[ARG0]])
  // CHECK-SAME: calls=[[CALLED_COMPUTATION]]
  %0 = "mhlo.async_start"(%arg0) {called_computation = @AsyncOp, execution_thread = "thread"} : (tensor<10xf32>) -> !mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>
  // CHECK: %[[UPDATE:.*]] = ((f32[10]), f32[20], s32[]) async-update(((f32[10]), f32[20], s32[]) %[[START]])
  %1 = "mhlo.async_update"(%0) : (!mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>) -> !mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>
  // CHECK: ROOT %{{.*}} = (f32[20]) async-done(((f32[10]), f32[20], s32[]) %[[UPDATE]])
  %2 = "mhlo.async_done"(%1) : (!mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>) -> tensor<20xf32>
  return %2 : tensor<20xf32>
}

// -----

// CHECK: HloModule
// CHECK: [[CALLED_COMPUTATION:%AsyncOp.*]] ([[ARG:.*]]: f32[10]) -> f32[20] {
func.func @AsyncOp(%arg0: tensor<10xf32>) -> tensor<20xf32>
  attributes {execution_thread = "thread"} {
  %1 = "mhlo.custom_call"(%arg0) {call_target_name = "bar"} : (tensor<10xf32>) -> tensor<20xf32>
  // CHECK: custom-call
  // CHECK-SAME: custom_call_target="bar"
  return %1 : tensor<20xf32>
}

// CHECK: ENTRY
func.func @main(%arg0: tensor<10xf32>) -> tensor<20xf32> {
  // CHECK: %[[ARG0:.*]] = f32[10] parameter(0)
  // CHECK: %[[START:.*]] = ((f32[10]), f32[20], s32[]) async-start(f32[10] %[[ARG0]]), async_execution_thread="thread", calls=[[CALLED_COMPUTATION]],
  // CHECK: %[[UPDATE:.*]] = ((f32[10]), f32[20], s32[]) async-update(((f32[10]), f32[20], s32[]) %[[START]])
  // CHECK: ROOT
  // CHECK-SAME: (f32[20]) async-done(((f32[10]), f32[20], s32[]) %[[UPDATE]])

  %0 = "mhlo.async_start"(%arg0) {called_computation = @AsyncOp, execution_thread="thread"} : (tensor<10xf32>) -> !mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) : (!mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>) -> !mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) : (!mhlo.async_bundle<tuple<tensor<10xf32>>, tensor<20xf32>, tensor<i32>>) -> tensor<20xf32>
  return %2 : tensor<20xf32>
}

// -----

// Breaking test case where tf2xla lowers to a send with a single manual
// sharding annotation on recv.

// CHECK: HloModule

// CHECK: ENTRY
func.func @main() -> tensor<1x2xf32> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "", outputs = "_retval0"}} {
  // CHECK:               %[[AFTER_ALL:.*]] = token[] after-all()
  // CHECK-NEXT:          %[[RECV:.*]] = (f32[1,2], u32[], token[]) recv(token[] %[[AFTER_ALL]]), channel_id=2, is_host_transfer=true,
  // CHECK-SAME{LITERAL}:   sharding={{manual}, {manual}, {manual}}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_1_retvals_htod_0"}
  // CHECK-NEXT:          %[[RECV_DONE:.*]] = (f32[1,2], token[]) recv-done((f32[1,2], u32[], token[]) %[[RECV]]), channel_id=2, is_host_transfer=true,
  // CHECK-SAME{LITERAL}:   sharding={{manual}, {manual}}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_1_retvals_htod_0"}
  // CHECK-NEXT:          ROOT %[[GET_TUPLE_0:.*]] = f32[1,2] get-tuple-element((f32[1,2], token[]) %[[RECV_DONE]]), index=0, sharding={manual}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_1_retvals_htod_0"}
  // CHECK-NEXT:          %[[GET_TUPLE_1:.*]] = token[] get-tuple-element((f32[1,2], token[]) %[[RECV_DONE]]), index=1, sharding={manual}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_1_retvals_htod_0"}
  %0 = mhlo.create_token : !mhlo.token
  %1:2 = "mhlo.recv"(%0) <{channel_handle = #mhlo.channel_handle<handle = 2, type = 3>, is_host_transfer = true}> {mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_rendezvous = "host_compute_channel_1_retvals_htod_0"}, mhlo.sharding = "\08\04"} : (!mhlo.token) -> (tensor<1x2xf32>, !mhlo.token)
  return %1#0 : tensor<1x2xf32>
}

// -----

// Check:
// - send has a 3 tuple sharding
// - send-done has a single sharding
// - recv has a 3 tuple sharding
// - recv-done has a 2 tuple sharding

// CHECK: HloModule

// CHECK: ENTRY
func.func @main(%arg0: tensor<1x2xi64>) -> tensor<1x2xi64> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "_arg0", outputs = "_retval0"}} {
  // CHECK:               %[[ARG0:.*]] = s64[1,2] parameter(0)
  // CHECK-NEXT:          %[[AFTER_ALL:.*]] = token[] after-all()
  // CHECK-NEXT:          %[[SEND:.*]] = (s64[1,2], u32[], token[]) send(s64[1,2] %[[ARG0]], token[] %[[AFTER_ALL]]), channel_id=3, is_host_transfer=true,
  // CHECK-SAME{LITERAL}:   sharding={{manual}, {manual}, {manual}}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_args_dtoh_0"}
  // CHECK-NEXT:          %[[SEND_DONE:.*]] = token[] send-done((s64[1,2], u32[], token[]) %[[SEND]]), channel_id=3, is_host_transfer=true, sharding={manual}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_args_dtoh_0"}
  // CHECK-NEXT:          %[[RECV:.*]] = (s64[1,2], u32[], token[]) recv(token[] %[[SEND_DONE]]), channel_id=4, is_host_transfer=true,
  // CHECK-SAME{LITERAL}:   sharding={{manual}, {manual}, {manual}}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_retvals_htod_0"}
  // CHECK-NEXT:          %[[RECV_DONE:.*]] = (s64[1,2], token[]) recv-done((s64[1,2], u32[], token[]) %[[RECV]]), channel_id=4, is_host_transfer=true,
  // CHECK-SAME{LITERAL}:   sharding={{manual}, {manual}}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_retvals_htod_0"}
  // CHECK-NEXT:          ROOT %[[GET_TUPLE_0:.*]] = s64[1,2] get-tuple-element((s64[1,2], token[]) %[[RECV_DONE]]), index=0, sharding={manual}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_retvals_htod_0"}
  // CHECK-NEXT:          %[[GET_TUPLE_1:.*]]  = token[] get-tuple-element((s64[1,2], token[]) %[[RECV_DONE]]), index=1, sharding={manual}, frontend_attributes={_xla_host_transfer_handler_name="tf_rendezvous",_xla_host_transfer_rendezvous="host_compute_channel_0_retvals_htod_0"}
  %0 = mhlo.create_token : !mhlo.token
  %1 = "mhlo.send"(%arg0, %0) <{channel_handle = #mhlo.channel_handle<handle = 3, type = 2>, is_host_transfer = true}> {mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_rendezvous = "host_compute_channel_0_args_dtoh_0"}, mhlo.sharding = "\08\04"} : (tensor<1x2xi64>, !mhlo.token) -> !mhlo.token
  %2:2 = "mhlo.recv"(%1) <{channel_handle = #mhlo.channel_handle<handle = 4, type = 3>, is_host_transfer = true}> {mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_rendezvous = "host_compute_channel_0_retvals_htod_0"}, mhlo.sharding = "\08\04"} : (!mhlo.token) -> (tensor<1x2xi64>, !mhlo.token)
  return %2#0 : tensor<1x2xi64>
}
