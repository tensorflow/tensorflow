// RUN: tf-tfrt-opt -pass-pipeline='tfrt-lower-tf-savedmodel' %s | FileCheck %s

// CHECK-NOT: tf_saved_model.semantics
module attributes {tf_saved_model.semantics} {

// CHECK-NOT: "tf_saved_model.global_tensor"
"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<i32>, value = dense<0> : tensor<i32>} : () -> ()
"tf_saved_model.global_tensor"() {is_mutable, sym_name = "z", type = tensor<i32>, value = dense<1> : tensor<i32>} : () -> ()

// CHECK-LABEL: func @test_basic
// CHECK-SAME: [[arg0:%.*]]: tensor<i32> {tf.resource_name = "y"},
// CHECK-SAME: [[arg1:%.*]]: tensor<i32> {tf.resource_name = "z"}
func @basic(
    %arg0: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y},
    %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @z})
      -> (tensor<i32> {tf_saved_model.index_path = ["r0"]}, tensor<i32> {tf_saved_model.index_path = ["r1"]})
  attributes {tf_saved_model.exported_names = ["test_basic"]} {

  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  // CHECK: return [[arg0]], [[arg1]]
  return %0, %1 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @then_branch
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
func private @then_branch(
    %arg0: tensor<!tf_type.resource<tensor<i32>>>,
    %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg0) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK: return [[arg0]]
  return %0 : tensor<i32>
}

// CHECK-LABEL: func private @else_branch
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
func private @else_branch(
    %arg0: tensor<!tf_type.resource<tensor<i32>>>,
    %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK: return [[arg1]]
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @test_if
// CHECK-SAME: [[arg0:%.*]]: tensor<i1>,
// CHECK-SAME: [[arg1:%.*]]: tensor<i32> {tf.resource_name = "y"},
// CHECK-SAME: [[arg2:%.*]]: tensor<i32> {tf.resource_name = "z"}
func @if(
    %arg0: tensor<i1> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y},
    %arg2: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @z})
      -> (tensor<i32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["test_if"]} {

  // CHECK: tf.If
  // CHECK-SAME: : (tensor<i1>, tensor<i32>, tensor<i32>)
  %0 = "tf.If"(%arg0, %arg1, %arg2) {then_branch = @then_branch, else_branch = @else_branch, is_stateless = false} : (tensor<i1>, tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  return %0 : tensor<i32>
}

// CHECK-LABEL: func @test_case
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>,
// CHECK-SAME: [[arg1:%.*]]: tensor<i32> {tf.resource_name = "y"},
// CHECK-SAME: [[arg2:%.*]]: tensor<i32> {tf.resource_name = "z"}
func @case(
    %arg0: tensor<i32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y},
    %arg2: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @z})
      -> (tensor<i32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["test_case"]} {

  // CHECK: tf.Case
  // CHECK-SAME: : (tensor<i32>, tensor<i32>, tensor<i32>)
  %0 = "tf.Case"(%arg0, %arg1, %arg2) {branches = [@then_branch, @else_branch], is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  return %0 : tensor<i32>
}

// CHECK-LABEL: func private @while_cond
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
func private @while_cond(
    %arg0: tensor<i32>,
    %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

// CHECK-LABEL: func private @while_body
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
// CHECK-SAME: -> (tensor<i32>, tensor<i32>)
func private @while_body(
    %arg0: tensor<i32>,
    %arg1: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) {
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %sum = "tf.AddV2"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>

  return %sum, %arg1 : tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>
}

// CHECK-LABEL: func @test_while
// CHECK-SAME: [[arg0:%.*]]: tensor<i32> {tf.resource_name = "z"}
func @while(
    %arg0: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @z})
      -> (tensor<i32> {tf_saved_model.index_path = ["r0"]}, tensor<i32> {tf_saved_model.index_path = ["r1"]})
  attributes {tf_saved_model.exported_names = ["test_while"]} {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // CHECK: tf.While
  // CHECK-SAME: (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %1, %2 = "tf.While"(%0, %arg0)
    {cond = @while_cond, body = @while_body, is_stateless = false} :
    (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>)

  // CHECK-NOT: tf.ReadVariableOp
  %3 = "tf.ReadVariableOp"(%2) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

  // CHECK: return {{%.*}}, {{%.*}} : tensor<i32>, tensor<i32>
  return %1, %3 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @batched_function
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
func private @batched_function(
    %arg0: tensor<i32>,
    %arg1: tensor<*x!tf_type.resource>) -> tensor<i32> {
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<i32>
  // CHECK: return {{%.*}} : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @test_batch_function
// CHECK-SAME: [[arg0:%.*]]: tensor<i32>, [[arg1:%.*]]: tensor<i32>
func @batch_function(
    %arg0: tensor<i32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf_type.resource<tensor<i32>>> {tf_saved_model.bound_input = @y})
      -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
      attributes {tf_saved_model.exported_names = ["test_batch_function"]} {
  // CHECK: tf.BatchFunction
  // CHECK-SAME: (tensor<i32>, tensor<i32>)
  %0 = "tf.BatchFunction"(%arg0, %arg1) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = dense<1> : vector<2xi32>, shared_name = "batch/"} : (tensor<i32>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  return %0 : tensor<i32>
}

}
