// RUN: tf-tfrt-opt -split-input-file -tfrt-sink-in-invariant-ops %s | FileCheck %s --dump-input=fail --dump-input-filter=all

module attributes {tf_saved_model.semantics} {

// Test sinks in var handle op to batch function.

// CHECK-LABEL: func private @batched_function
// CHECK: arg1
func.func private @batched_function(%arg0: tensor<1x3xf32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<1x3xf32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<1x3xf32>
  %1 = "tf.AddV2"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["r"]}) 
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK: tf.VarHandleOp
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
 
  // CHECK: "tf.BatchFunction"(%arg0, %0)
  // CHECK: operand_segment_sizes = array<i32: 1, 1>
  %1 = "tf.BatchFunction"(%arg0, %0) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test sinks in const op to batch function.

// CHECK-LABEL: func private @batched_function
// CHECK: arg1
func.func private @batched_function(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK: tf.Const
  %1 = "tf.AddV2"(%arg0, %arg1) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK: [[handle:%.*]] = "tf.Const"()
  %0 = "tf.Const"() {device = "/CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.BatchFunction"(%arg0, [[handle]])
  // CHECK-SAME: operand_segment_sizes = array<i32: 1, 1>
  %1 = "tf.BatchFunction"(%arg0, %0) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test sink in multiple invariant ops.

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<!tf_type.resource<tensor<1x3xf32>>>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK: [[handle1:%.*]] = "tf.VarHandleOp"() {{{.*}}, shared_name = "variable1"}
  // CHECK: [[handle2:%.*]] = "tf.VarHandleOp"() {{{.*}}, shared_name = "variable2"}
  // CHECK: "tf.ReadVariableOp"([[handle1]])
  // CHECK: "tf.ReadVariableOp"([[handle2]])
  %0 = "tf.ReadVariableOp"(%arg0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %1 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32> 
  %2 = "tf.AddV2"(%0, %1) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %3 = "tf.Identity"(%2) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %3 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["r"]}) 
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK: tf.VarHandleOp
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable1"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable2"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  // CHECK: "tf.BatchFunction"(%0, %1)
  %2 = "tf.BatchFunction"(%0, %1) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test sinks in var handle op that used by control flow ops.

// CHECK-LABEL: func private @some_func
func.func private @some_func(
    %arg: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK: tf.VarHandleOp
  // CHECK: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg) {device = "cpu"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func private @some_other_func
func.func private @some_other_func(
    %arg: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %0 = "tf.ReadVariableOp"(%arg) {device = "cpu"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @sink_in_stateful_call
func.func @sink_in_stateful_call(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_sink_in_stateful_call"]} {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK: "tf.StatefulPartitionedCall"([[handle]])
  %x = "tf.StatefulPartitionedCall"(%handle) {device = "/CPU:0", config = "", config_proto = "", executor_type = "", f = @some_func} : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>)
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

// CHECK-LABEL: func @sink_in_if
func.func @sink_in_if(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_sink_in_if"]} {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK: [[cond:%.*]] = "tf.Const"()
  %cond = "tf.Const"() {device = "/CPU:0", value = dense<true> : tensor<i1>} : () -> tensor<i1>
  // CHECK: "tf.If"([[cond]], [[handle]])
  %x = "tf.If"(%cond, %handle) {then_branch = @some_other_func, else_branch = @some_other_func, is_stateless = false} : (tensor<i1>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test doesn't sink in to the callee that invoked by multiple callers.

// CHECK: func private @some_func([[arg0:.+]]: tensor<!tf_type.resource<tensor<i32>>>)
func.func private @some_func(%arg0: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg0) {device = "cpu"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @sink_in_stateful_call
func.func @sink_in_stateful_call(%arg0: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_sink_in_stateful_call"]} {
  // CHECK: tf.VarHandleOp
  %0 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK: "tf.StatefulPartitionedCall"(%0)
  %1 = "tf.StatefulPartitionedCall"(%0) {device = "/CPU:0", config = "", config_proto = "", executor_type = "", f = @some_func} : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>)
  %2 = "tf.AddV2"(%arg0, %1) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

// CHECK-LABEL: func @sink_in_if
func.func @sink_in_if(%arg0: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_sink_in_if"]} {
  // CHECK: tf.VarHandleOp
  %0 = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %cst = "tf.Const"() {device = "/CPU:0", value = dense<true> : tensor<i1>} : () -> tensor<i1>
  // CHECK: "tf.If"(%cst, %0)
  %1 = "tf.If"(%cst, %0) {then_branch = @some_func, else_branch = @some_func, is_stateless = false} : (tensor<i1>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf.AddV2"(%arg0, %1) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test doesn't sink in var handle op + read variable op. Consider implement when we see it from production.

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.ReadVariableOp
  %1 = "tf.AddV2"(%arg0, %arg1) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["r"]}) 
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: "tf.BatchFunction"(%arg0, %1)
  %2 = "tf.BatchFunction"(%arg0, %1) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test sinks in var handle op if it's used by one callee, and also by read only ops in the current funciton.

// CHECK-LABEL: func private @batched_function
// CHECK: arg1
func.func private @batched_function(%arg0: tensor<1x3xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK: tf.VarHandleOp
  // CHECK: tf.ReadVariableOp
  %1 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %2 = "tf.AddV2"(%arg0, %1) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %3 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %2 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["r"]}) 
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"()
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: "tf.BatchFunction"(%arg0, [[handle]])
  // CHECK-SAME: operand_segment_sizes = array<i32: 1, 1>
  %2 = "tf.BatchFunction"(%arg0, %0) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

}