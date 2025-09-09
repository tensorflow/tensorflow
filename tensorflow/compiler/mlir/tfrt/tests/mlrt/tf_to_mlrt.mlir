// RUN: tf-tfrt-opt -split-input-file -tf-to-mlrt %s | FileCheck %s

// CHECK-LABEL: @main_stream_0
// CHECK-SAME: ([[input0:%.*]]: !tf_mlrt.tensor, [[promise_b:%.*]]: !mlrt.promise)
func.func @main_stream_0(%input0: tensor<i32>, %promise_b: !mlrt.promise) {
  %const = "tf.Const"() {__op_key = 0 : i32, value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[a:%.*]] = tf_mlrt.executeop([[input0]],
  // CHECK-SAME: AddV2
  %a = "tf.AddV2"(%input0, %const) {__op_key = 1: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[b:%.*]] = tf_mlrt.executeop([[a]])
  // CHECK-SAME: Abs
  %b = "tf.Abs"(%a) {__op_key = 2 : i32}: (tensor<i32>) -> tensor<i32>
  // CHECK: tf_mlrt.promise [[promise_b]], [[b]]
  "tf_mlrt.tf_promise"(%promise_b, %b) : (!mlrt.promise, tensor<i32>) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: @main_stream_1
// CHECK-SAME: ([[input1:%.*]]: !tf_mlrt.tensor, [[promise_c:%.*]]: !mlrt.promise, [[promise_d:%.*]]: !mlrt.promise)
func.func @main_stream_1(%input1: tensor<i32>, %promise_c: !mlrt.promise, %promise_d: !mlrt.promise) {
  %const = "tf.Const"() {__op_key = 3 : i32, value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[c:%.*]] = tf_mlrt.executeop([[input1]],
  // CHECK-SAME: Sub
  %c = "tf.Sub"(%input1, %const) {__op_key = 4: i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: tf_mlrt.promise [[promise_c]], [[c]]
  "tf_mlrt.tf_promise"(%promise_c, %c) : (!mlrt.promise, tensor<i32>) -> ()
  // CHECK: [[d:%.*]] = tf_mlrt.executeop([[c]])
  // CHECK-SAME: Abs
  %d = "tf.Abs"(%c) {__op_key = 5: i32}: (tensor<i32>) -> tensor<i32>
  // CHECK: tf_mlrt.promise [[promise_d]], [[d]]
  "tf_mlrt.tf_promise"(%promise_d, %d) : (!mlrt.promise, tensor<i32>) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: @main
// CHECK-SAME: ([[input0:%.*]]: !tf_mlrt.tensor, [[input1:%.*]]: !tf_mlrt.tensor)
func.func @main(%input0: tensor<i32>, %input1: tensor<i32>) -> tensor<i32> {
  // CHECK: [[promises:%.*]]:3, [[futures:%.*]]:3 = "tf_mlrt.allocate_futures"
  // CHECK-SAME: num_futures = 3
  %promise_b, %promise_c, %promise_d, %future_b, %future_c, %future_d =
    "tf_mlrt.allocate_futures"()
    {num_futures = 3 : i32, resultSegmentSizes = array<i32: 3, 3>} : () ->
    (!mlrt.promise, !mlrt.promise, !mlrt.promise,
     !mlrt.future, !mlrt.future, !mlrt.future)

  // CHECK: [[handle_0:%.*]] = mlrt.async([[input0]], [[promises]]#0)
  // CHECK-SAME: callee = @main_stream_0
  %handle_0 = mlrt.async(%input0, %promise_b)
    {callee = @main_stream_0} :
    (tensor<i32>, !mlrt.promise) -> !mlrt.async_handle
  // CHECK: [[handle_1:%.*]] = mlrt.async([[input1]], [[promises]]#1, [[promises]]#2)
  // CHECK-SAME: callee = @main_stream_1
  %handle_1 = mlrt.async(%input1, %promise_c, %promise_d)
    {callee = @main_stream_1} :
    (tensor<i32>, !mlrt.promise, !mlrt.promise) -> !mlrt.async_handle

  %const = "tf.Const"() {__op_key = 6: i32, value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[e:%.*]] = tf_mlrt.executeop([[input1]],
  // CHECK-SAME: Mul
  %e = "tf.Mul"(%input1, %const) {__op_key = 7: i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[c:%.*]] = tf_mlrt.await [[futures]]#1
  %c = "tf_mlrt.tf_await"(%future_c) : (!mlrt.future) ->tensor<i32>
  // CHECK: [[f:%.*]] = tf_mlrt.executeop([[e]], [[c]])
  // CHECK-SAME: Div
  %f = "tf.Div"(%e, %c) {__op_key = 8: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK: [[b:%.*]] = tf_mlrt.await [[futures]]#0
  %b = "tf_mlrt.tf_await"(%future_b) : (!mlrt.future) ->tensor<i32>
  // CHECK: [[d:%.*]] = tf_mlrt.await [[futures]]#2
  %d = "tf_mlrt.tf_await"(%future_d) : (!mlrt.future) ->tensor<i32>

  // CHECK: [[result:%.*]] = tf_mlrt.executeop([[b]], [[d]], [[f]])
  // CHECK-SAME: AddN
  %result = "tf.AddN"(%b, %d, %f) {__op_key = 9: i32}: (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK: mlrt.await_handle [[handle_0]]
  // CHECK: mlrt.await_handle [[handle_1]]
  mlrt.await_handle %handle_0
  mlrt.await_handle %handle_1

  // CHECK: return [[result]]
  return %result : tensor<i32>
}

// -----

// Test lowering tf.If

func.func @then(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  return %x: tensor<i32>
}

func.func @else(%x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  return %y: tensor<i32>
}

// CHECK-LABEL: func @main
// CHECK-SAME: ([[cond_tensor:%.*]]: !tf_mlrt.tensor, [[x:%.*]]: !tf_mlrt.tensor, [[y:%.*]]: !tf_mlrt.tensor)
// CHECK: [[cond:%.*]] = tf_mlrt.predicate [[cond_tensor]]
// CHECK: [[z:%.*]] = mlrt.cond [[cond]] @then @else([[x]], [[y]])
// CHECK: return [[z]]
func.func @main(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> tensor<i32> {
  %z = "tf.If"(%cond, %x, %y) {then_branch = @then, else_branch = @else, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  return %z: tensor<i32>
}

// -----

// Test lowering AsyncOpKernel

// CHECK-LABEL: func @main
func.func @main(%x: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK: [[y_future:%.*]] = tf_mlrt.async_executeop
  %y = "tf.TestAsyncIdentity"(%x) {__op_key = 0: i32, T = i32} : (tensor<i32>) -> tensor<i32>
  // CHECK: [[z:%.*]] = tf_mlrt.executeop
  %z = "tf.Identity"(%x) {__op_key = 1: i32}: (tensor<i32>) -> tensor<i32>
  // CHECK: [[y:%.*]] = tf_mlrt.await [[y_future]]
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %w = "tf.AddV2"(%y, %z) {__op_key = 2: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %u = "tf.AddV2"(%y, %z) {__op_key = 3: i32} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %v = "tf.AddV2"(%y, %z) {__op_key = 4: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %w, %u, %v : tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// Test lowering BatchFunction op.

func.func @batched_function(%x: tensor<?xi32>) -> tensor<?xi32> {
  return %x : tensor<?xi32>
}

// CHECK-LABEL: func @main
func.func @main(%x: tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) {
  // CHECK: [[y_future:%.*]] = tf_mlrt.batch_function
  // CHECK-SAME: f = @batched_function
  // CHECK-SAME: \22batch_function\22
  %y = "tf.BatchFunction"(%x) {
    allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64,
    batching_queue = "", container = "", device = "/device:CPU:0",
    enable_large_batch_splitting = false, f = @batched_function,
    max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64,
    num_batch_threads = 1 : i64, operandSegmentSizes = array<i32: 1, 0>,
    shared_name = "batch_function"
  } : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: [[z:%.*]] = tf_mlrt.executeop
  %z = "tf.Identity"(%x) {__op_key = 0: i32} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK: [[y:%.*]] = tf_mlrt.await [[y_future]]
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %w = "tf.AddV2"(%y, %z) {__op_key = 1: i32}: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %u = "tf.AddV2"(%y, %z) {__op_key = 2: i32}: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: tf_mlrt.executeop([[y]]
  %v = "tf.AddV2"(%y, %z) {__op_key = 3: i32}: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %w, %u, %v : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
}

// -----

// Test node names are preserved.

// CHECK-LABEL: func @main
func.func @main(%x: tensor<i32>) -> tensor<i32> {
  // CHECK: tf_mlrt.executeop
  // CHECK-SAME: name: \22name_loc/AddV2_0\22
  %y = "tf.AddV2"(%x, %x) {__op_key = 0: i32} : (tensor<i32>, tensor<i32>) -> tensor<i32> loc("name_loc:AddV2")
  // CHECK: tf_mlrt.executeop
  // CHECK-SAME: name: \22fused_loc/AddV2_1\22
  %z = "tf.AddV2"(%y, %x) {__op_key = 1: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32> loc(fused["fused_loc:", "AddV2"])
  // CHECK: tf_mlrt.executeop
  // CHECK-SAME: name: \22AddV2_2\22
  %w = "tf.AddV2"(%z, %x) {__op_key = 2: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %z : tensor<i32>
}

// -----

// Test function name canonicalization

// CHECK-LABEL: func @__inference_pruned_35
func.func @__inference_pruned_35() -> tensor<!tf_type.variant> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "flatmapdataset__4_RetVal"}} {
  %0 = "tf.Const"() {__op_key = 0: i32, device = "/device:CPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %1 = "tf.Const"() {__op_key = 1: i32, device = "/device:CPU:0", value = dense<5> : tensor<i64>} : () -> tensor<i64>
  %2 = "tf.Const"() {__op_key = 2: i32, device = "/device:CPU:0", value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %3 = "tf.RangeDataset"(%0, %1, %2) {__op_key = 3: i32, device = "/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<!tf_type.variant>
  // CHECK: tf_mlrt.executeop{{.*}}op: \22FlatMapDataset\22
  // CHECK-SAME: \22__inference_Dataset_flat_map_lambda_19\22
  %4 = "tf.FlatMapDataset"(%3) {__op_key = 4: i32, Targuments = [], device = "/device:CPU:0", f = @__inference_Dataset_flat_map_lambda_190, output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  func.return %4 : tensor<!tf_type.variant>
}
// CHECK-LABEL: __inference_Dataset_flat_map_lambda_190
func.func private @__inference_Dataset_flat_map_lambda_190(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}) -> tensor<!tf_type.variant> attributes {tf._original_func_name = "__inference_Dataset_flat_map_lambda_19", tf._tf_data_function = true, tf.signature.is_stateful} {
  %0 = "tf.Const"() {__op_key = 5: i32, device = "/device:CPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %1 = "tf.Const"() {__op_key = 6: i32,device = "/device:CPU:0", value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %2 = "tf.Const"() {__op_key = 7: i32,device = "/device:CPU:0", value = dense<5> : tensor<i64>} : () -> tensor<i64>
  %3 = "tf.RangeDataset"(%0, %2, %1) {__op_key = 8: i32, device = "/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<!tf_type.variant>
  // CHECK: tf_mlrt.executeop{{.*}}op: \22MapDataset\22
  // CHECK-SAME: \22__inference_Dataset_map_lambda_16\22
  %4 = "tf.MapDataset"(%3) {__op_key = 9: i32, device = "/device:CPU:0", f = @__inference_Dataset_map_lambda_160, f._tf_data_function = true, output_shapes = [#tf_type.shape<>], output_types = [i64], preserve_cardinality = true, use_inter_op_parallelism = true, metadata = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  %5 = "tf.Identity"(%4) {__op_key = 10: i32, device = "/device:CPU:0"} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  func.return %5 : tensor<!tf_type.variant>
}
// CHECK-LABEL: __inference_Dataset_map_lambda_160
func.func private @__inference_Dataset_map_lambda_160(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}) -> tensor<i64> attributes {tf._tf_data_function = true} {
  %0 = "tf.Const"() {__op_key = 11: i32, device = "/device:CPU:0", value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %1 = "tf.Mul"(%arg0, %0) {__op_key = 12: i32, device = "/device:CPU:0"} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %2 = "tf.Identity"(%1) {__op_key = 13: i32, device = "/device:CPU:0"} : (tensor<i64>) -> tensor<i64>
  func.return %2 : tensor<i64>
}

// -----

// Test while conversion

// CHECK-LABEL: func @while_cond_lt9
// CHECK-SAME: ([[arg0:%.*]]: !tf_mlrt.tensor) -> !tf_mlrt.tensor
func.func @while_cond_lt9(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {__op_key = 0: i32, device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {__op_key = 1: i32, device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// CHECK-LABEL: func @while_body_add2
// CHECK-SAME: ([[arg0:%.*]]: !tf_mlrt.tensor) -> !tf_mlrt.tensor
func.func @while_body_add2(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {__op_key = 2: i32, device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {__op_key = 3: i32, device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @while_test()
// CHECK-SAME: -> !tf_mlrt.tensor
func.func @while_test() -> (tensor<i32>) {
  // CHECK: [[CONST:%.*]] = tf_mlrt.constop
  %0 = "tf.Const"() {__op_key = 4: i32, device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[pred_res:%.*]] = call @"while_cond_lt9/tf_mlrt_predicate"([[CONST]]) : (!tf_mlrt.tensor) -> i1
  // CHECK: [[while_res:%.*]]:2 = mlrt.while
  // CHECK-SAME: @"while_body_add2/tf_mlrt_body"([[CONST]])
  // CHECK-SAME: (!tf_mlrt.tensor) -> (!tf_mlrt.tensor, i1)
  %1 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: return [[while_res]]#0 : !tf_mlrt.tensor
  func.return %1 : tensor<i32>
}
// CHECK: func @"while_body_add2/tf_mlrt_body"([[arg:%.*]]: !tf_mlrt.tensor) -> (!tf_mlrt.tensor, i1)
// CHECK: [[body_res:%.*]] = call @while_body_add2([[arg]]) : (!tf_mlrt.tensor) -> !tf_mlrt.tensor
// CHECK: [[pred_res:%.*]] = call @"while_cond_lt9/tf_mlrt_predicate"([[body_res]]) : (!tf_mlrt.tensor) -> i1
// CHECK: return [[body_res]], [[pred_res]] : !tf_mlrt.tensor, i1

// CHECK: func @"while_cond_lt9/tf_mlrt_predicate"([[arg:%.*]]: !tf_mlrt.tensor) -> i1
// CHECK: [[cond_res:%.*]] = call @while_cond_lt9([[arg]]) : (!tf_mlrt.tensor) -> !tf_mlrt.tensor
// CHECK: [[bool_res:%.*]] = tf_mlrt.predicate [[cond_res]]
// CHECK: return [[bool_res]] : i1

// CHECK-LABEL: func @multi_while_test
func.func @multi_while_test() -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Const"() {__op_key = 5: i32, device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {__op_key = 6: i32, device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[pred_0:%.*]] = call @"while_cond_lt9/tf_mlrt_predicate"
  // CHECK: mlrt.while [[pred_0]] @"while_body_add2/tf_mlrt_body"
  // CHECK: [[pred_1:%.*]] = call @"while_cond_lt9/tf_mlrt_predicate"
  // CHECK: mlrt.while [[pred_1]] @"while_body_add2/tf_mlrt_body"
  %2 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  %3 = "tf.While"(%1) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  func.return %2, %3 : tensor<i32>, tensor<i32>
}

// -----

// Test async output to function is converted

// CHECK-LABEL: @serving_default_stream_1
// CHECK-SAME: !mlrt.future
func.func private @serving_default_stream_1(%arg0: tensor<i32>) {
  // CHECK: [[tensor:%.*]] = tf_mlrt.await
  // CHECK: tf_mlrt.executeop([[tensor]])
  %0 = "tf.StringFormat"(%arg0) {__op_key = 0: i32, device = "/job:localhost/replica:0/task:0/device:CPU:0", placeholder = "{}", strtemplate = "%s", summarize = 3 : i64, template = "Outside compiled {}"} : (tensor<i32>) -> tensor<!tf_type.string>
  "tf.PrintV2"(%0) {__op_key = 1: i32, device = "/job:localhost/replica:0/task:0/device:CPU:0", end = "\0A", output_stream = "stderr"} : (tensor<!tf_type.string>) -> ()
  return
}

func.func @callee(%arg: tensor<i32>) -> (tensor<i32>) {
  func.return %arg: tensor<i32>
}

// CHECK-LABEL: @executeop_input
func.func @executeop_input(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[async_out:%.*]] = tf_mlrt.batch_function
  %2 = "tf.BatchFunction"(%arg0) {device = "/device:CPU:0", allowed_batch_sizes = [64], batch_timeout_micros = 1 : i64, batching_queue = "", container = "", f = @callee, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>, shared_name = ""} : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: mlrt.async([[async_out]]) {{.*}} : (!mlrt.future)
  %3 = mlrt.async(%2) {callee = @serving_default_stream_1} : (tensor<i32>) -> !mlrt.async_handle
  // CHECK: mlrt.await_handle
  mlrt.await_handle %3
  // CHECK: return
  // CHECK-SAME: !tf_mlrt.tensor
  func.return %2 : tensor<i32>
}

// -----

// Support pre-assigned op_key

// CHECK-LABEL: @main
// CHECK-SAME: ([[input0:%.*]]: !tf_mlrt.tensor, [[promise_b:%.*]]: !mlrt.promise)
func.func @main(%input0: tensor<i32>, %promise_b: !mlrt.promise) {
  %const = "tf.Const"() {__op_key = 0 : i32, value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[a:%.*]] = tf_mlrt.executeop([[input0]],
  // CHECK-SAME: AddV2
  // CHECK-SAME: op_key = 1
  // CHECK-NOT: __op_key
  %a = "tf.AddV2"(%input0, %const) {__op_key = 1: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[b:%.*]] = tf_mlrt.executeop([[a]])
  // CHECK-SAME: Abs
  // CHECK-SAME: op_key = 2
  // CHECK-NOT: __op_key
  %b = "tf.Abs"(%a) {__op_key = 2: i32 }: (tensor<i32>) -> tensor<i32>
  // CHECK: tf_mlrt.promise [[promise_b]], [[b]]
  "tf_mlrt.tf_promise"(%promise_b, %b) : (!mlrt.promise, tensor<i32>) -> ()
  // CHECK: return
  return
}

// -----

// Test future as input to promise

// CHECK-LABEL: func @main_stream_0
func.func @main_stream_0(%x: tensor<i32>, %p: !mlrt.promise) -> () {
  // CHECK: [[y_future:%.*]] = tf_mlrt.async_executeop
  %y = "tf.TestAsyncIdentity"(%x) {__op_key = 0: i32, T = i32} : (tensor<i32>) -> tensor<i32>
  // CHECK: tf_mlrt.promise_future
  // CHECK-SAME: [[y_future]]
  "tf_mlrt.tf_promise"(%p, %y): (!mlrt.promise, tensor<i32>) -> ()
  return
}

// CHECK-LABEL: @main
// CHECK-SAME: ([[input0:%.*]]: !tf_mlrt.tensor)
func.func @main(%input0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[promises:%.*]], [[futures:%.*]] = "tf_mlrt.allocate_futures"
  // CHECK-SAME: num_futures = 1
  %promise_b, %future_b = "tf_mlrt.allocate_futures"()
    {num_futures = 1 : i32, resultSegmentSizes = array<i32: 1, 1>} : () ->
    (!mlrt.promise, !mlrt.future)

  // CHECK: [[handle_0:%.*]] = mlrt.async([[input0]], [[promises]])
  // CHECK-SAME: callee = @main_stream_0
  %handle_0 = mlrt.async(%input0, %promise_b)
    {callee = @main_stream_0} :
    (tensor<i32>, !mlrt.promise) -> !mlrt.async_handle

  // CHECK: [[const:%.*]]  = tf_mlrt.const
  %const = "tf.Const"() {__op_key = 1: i32, value = dense<2> : tensor<i32>} : () -> tensor<i32>

  // CHECK: [[b:%.*]] = tf_mlrt.await [[futures]]
  %b = "tf_mlrt.tf_await"(%future_b) : (!mlrt.future) ->tensor<i32>

  // CHECK: [[result:%.*]] = tf_mlrt.executeop([[b]], [[const]])
  // CHECK-SAME: AddV2
  %result = "tf.AddV2"(%b, %const) {__op_key = 2: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>

  // CHECK: mlrt.await_handle [[handle_0]]
  mlrt.await_handle %handle_0

  // CHECK: return [[result]]
  return %result : tensor<i32>
}

// -----

// Test lowering of tf call ops

// CHECK-LABEL: @callee
func.func @callee(%arg0: tensor<i32>) -> (tensor<i32>) {
  func.return %arg0: tensor<i32>
}

// CHECK-LABEL: func @call_test
func.func @call_test(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %0 = "tf.Add"(%arg0, %arg0) {__op_key = 0, device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[results_0:%.*]] = call @callee(
  // CHECK-SAME: (!tf_mlrt.tensor) -> !tf_mlrt.tensor
  %1 = "tf.StatefulPartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK-NEXT: [[results_1:%.*]] = call @callee(
  // CHECK-SAME: (!tf_mlrt.tensor) -> !tf_mlrt.tensor
  %2 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK-NEXT: [[results_2:%.*]] = call @callee(
  // CHECK-SAME: (!tf_mlrt.tensor) -> !tf_mlrt.tensor
  %3 = "tf.LegacyCall"(%0) {f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[results_0]], [[results_1]], [[results_2]]
  func.return %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: @branch0
func.func @branch0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {__op_key = 1, device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @branch1
func.func @branch1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {__op_key = 2, device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tf.Add" (%arg0, %0) {__op_key = 3, device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: func @case_test
// CHECK-SAME: ([[tf_idx:%.*]]: !tf_mlrt.tensor, [[branch_arg0:%.*]]: !tf_mlrt.tensor, [[branch_arg1:%.*]]: !tf_mlrt.tensor)
func.func @case_test(%arg0: tensor<i32>, %arg1: tensor<f32>,  %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[idx:%.*]] = tf_mlrt.tensor_to_int32 [[tf_idx]]
  // CHECK-NEXT: [[out:%.*]] = mlrt.case [[idx]] [@branch0, @branch1]([[branch_arg0]], [[branch_arg1]])
  %0 = "tf.Case"(%arg0, %arg1, %arg2) {_lower_using_switch_merge = true, branches = [@branch0, @branch1], is_stateless = true} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Test await is added for unused futures

// CHECK-LABEL: func @unused_future_arg
// CHECK-SAME: ({{%.*}}: !tf_mlrt.tensor, [[unused:%.*]]: !mlrt.future)
func.func @unused_future_arg(%x: tensor<i32>, %unused: !mlrt.future) -> tensor<i32> {
  // CHECK: mlrt.await_all_control [[unused]]
  return %x : tensor<i32>
}

// CHECK-LABEL: func @unused_future
func.func @unused_future(%x: tensor<i32>) -> tensor<i32> {
  // CHECK: [[unused:%.*]] = tf_mlrt.async_executeop
  %unused = "tf.TestAsyncIdentity"(%x) {__op_key = 0: i32, T = i32} : (tensor<i32>) -> tensor<i32>
  // CHECK: mlrt.await_all_control [[unused]]
  return %x : tensor<i32>
}

// -----

// Test for XlaLaunch

func.func private @xla_func_0(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) {__op_key = 0: i32} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1 : tensor<1x3xf32>
}

// CHECK-LABEL: func @xla_func
func.func @xla_func(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {__op_key = 1: i32, device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {__op_key = 2: i32, device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: tf_mlrt.executeop
  // CHECK: tf_mlrt.async_executeop{{.*}}op: \22XlaLaunch\22\0A
  // CHECK: tf_mlrt.await
  // CHECK: return
  // CHECK-SAME: !tf_mlrt.tensor
  %2 = "tf.XlaLaunch"(%arg0, %1) {__op_key = 3: i32, _noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_0, operandSegmentSizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// Test lowering of IfrtLoadVariableOp

// CHECK-LABEL: func @ifrt_load_variable_test
func.func @ifrt_load_variable_test() -> () {
  // CHECK: [[HANDLE:%.*]] = tf_mlrt.executeop()
  // CHECK-SAME:  VarHandleOp
  %0 = "tf.VarHandleOp"() {__op_key = 1: i32, device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  // CHECK-NEXT: "tf_mlrt.ifrt_load_variable"([[HANDLE]])
  // CHECK-SAME: used_by_host = true
  %1, %2 = "tf_mlrt.tf_ifrt_load_variable"(%0) {used_by_host = true, __op_key = 2: i32, device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> (tensor<!tf_type.string>, !mlrt.future)
  // CHECK-NEXT: mlrt.await_all_control
  // CHECK-NEXT: return
  func.return
}

// -----

// Test lowering of IfrtRestoreVariableOp

// CHECK-LABEL: func @ifrt_restore_variable_test
func.func @ifrt_restore_variable_test() -> () {
  // CHECK-NEXT: [[PREFIX:%.*]] = tf_mlrt.constop
  %cst = "tf.Const"() {__op_key = 0: i32, value = dense<"restore_ariables"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  // CHECK-NEXT: [[SLICE:%.*]] = tf_mlrt.constop
  %cst_0 = "tf.Const"()  {__op_key = 1: i32, value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  // CHECK-NEXT: [[NAME:%.*]] = tf_mlrt.constop
  %cst_1 = "tf.Const"()  {__op_key = 2: i32, value = dense<["y"]> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  // CHECK-NEXT: [[HANDLE:%.*]] = tf_mlrt.executeop
  %handle = "tf.VarHandleOp"() {__op_key = 3: i32, container = "x", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  // CHECK-NEXT: "tf_mlrt.ifrt_restore_variable"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLE]]) <{restored_dtypes = [f32], truncate_in_cast = array<i1: true>}>
  "tf.IfrtRestoreVariableOp"(%cst, %cst_1, %cst_0, %handle) {restored_dtypes = [f32], truncate_in_cast = array<i1: true>} : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<!tf_type.resource<tensor<3x1xf32>>>) -> ()
  // CHECK-NEXT: return
  func.return
}

// -----

// Test lowering of tf.IfrtResourceDeserializeOp to tf_mlrt.ifrt_resource_deserialize

// CHECK-LABEL: func @ifrt_resource_deserialize_test
func.func @ifrt_resource_deserialize_test(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  %input_dir = "tf.Const"() { value = dense<"some/path"> : tensor<!tf_type.string> } : () -> tensor<!tf_type.string>
  // CHECK: "tf_mlrt.ifrt_resource_deserialize"(%arg0, %{{.*}}) <{tensor_name = "my_tensor"}>
  "tf.IfrtResourceDeserialize"(%arg0, %input_dir) {require_matching_crc = false, tensor_name = "my_tensor"} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.string>) -> ()
  func.return
}

