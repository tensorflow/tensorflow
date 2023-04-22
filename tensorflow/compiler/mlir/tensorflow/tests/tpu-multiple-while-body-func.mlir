// RUN: tf-opt %s -tf-tpu-bridge 2>&1 | FileCheck %s
// RUN: tf-opt %s -tf-tpu-bridge-v1 2>&1 | FileCheck %s

// This test verifies there is no warning about shape inference failure in TPU
// bridge in handling multiple usage of the same function.

// Since it is possible that this warning may become an error in the future,
// only check the message content here.

// CHECK-NOT: expected control flow function {{.*}} to have exactly 1 use, found 2

"module"() ( {
  "func"() ( {
  ^bb0(%arg0: tensor<i32>):  // no predecessors
    %1617 = "tf.While"(%arg0) {_lower_using_switch_merge = true, _num_original_outputs = 7 : i64, _read_only_resource_inputs = [], body = @main_while_body_4225150, cond = @main_while_cond_4225140, device = "", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>) -> (tensor<i32>)
    "std.return"() : () -> ()
  }) {sym_name = "__inference_wrapped_function_4260250", sym_visibility = "private", tf._input_shapes = [#tf.shape<>], tf.signature.is_stateful, type = (tensor<i32>) -> ()} : () -> ()
  "func"() ( {
  ^bb0(%arg0: tensor<i32>):  // no predecessors
    %1617 = "tf.While"(%arg0) {_lower_using_switch_merge = true, _num_original_outputs = 7 : i64, _read_only_resource_inputs = [], body = @main_while_body_4225150, cond = @main_while_cond_4225140, device = "", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>) -> (tensor<i32>)
    "std.return"() : () -> ()
  }) {sym_name = "__inference_wrapped_function_4260250_0", sym_visibility = "private", tf._input_shapes = [#tf.shape<>], tf.signature.is_stateful, type = (tensor<i32>) -> ()} : () -> ()
  "func"() ( {
  ^bb0(%arg0: tensor<i32>):  // no predecessors
    "std.return"(%arg0) : (tensor<i32>) -> ()
  }) {sym_name = "main_while_body_4225150", sym_visibility = "private", tf._input_shapes = [#tf.shape<>], tf.signature.is_stateful, type = (tensor<i32>) -> (tensor<i32>)} : () -> ()
  "func"() ( {
  ^bb0(%arg0: tensor<i32>):  // no predecessors
    %0 = "tf.Less"(%arg0, %arg0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
    %1 = "tf.Less"(%arg0, %arg0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
    %2 = "tf.LogicalAnd"(%0, %1) {device = ""} : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
    "std.return"(%2) : (tensor<*xi1>) -> ()
  }) {sym_name = "main_while_cond_4225140", sym_visibility = "private", tf._input_shapes = [#tf.shape<>], type = (tensor<i32>) -> tensor<*xi1>} : () -> ()
}) {tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"], tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 651 : i32}, tf_saved_model.semantics} : () -> ()
