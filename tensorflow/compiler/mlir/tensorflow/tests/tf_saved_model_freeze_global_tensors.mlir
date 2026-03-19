// RUN: tf-opt -verify-diagnostics -tf-saved-model-freeze-global-tensors -split-input-file %s | FileCheck %s

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: Basic freezing.

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  // CHECK: func @f()
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}>
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: Sanity check handling of non-bound inputs.
  // The pass shouldn't do anything in this case.

  // CHECK: func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>>  {tf_saved_model.index_path = [0]})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: Fail if mutable global tensors are found.

  // expected-error @+1 {{is not immutable}}
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    func.return
  }

}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: success if bound input user's only none ReadVariableOp instance
  // is call.

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<21.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @f_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @g_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @g_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() <{value = dense<2.100000e+01> : tensor<f32>}>
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // Test case: success if bound input user's only none ReadVariableOp instance
  // is call with read inside function

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<32.0> : tensor<f32> } : () -> ()

  func.func @g(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["g"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @g_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @g_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    // CHECK: "tf.Const"() <{value = dense<3.200000e+01> : tensor<f32>}>
    func.return
  }
}

// -----

module attributes {tf_saved_model.semantics} {

  // Test case: fail if bound input user's only none ReadVariableOp instance
  // is call with write inside function

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<f32>>>) -> ()
    func.return
  }

  func.func private @f_callee(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
    %0 = "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // expected-error @+1 {{immutable bound input}}
    "tf.AssignAddVariableOp"(%arg0, %0) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    func.return
  }
}

// -----

// expected-error @+1 {{could not freeze all global tensors in the module}}
module attributes {tf_saved_model.semantics} {

  // Test case: Fail if some global tensor ops remain

 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()
 "tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<2.0> : tensor<f32> } : () -> ()

  func.func @f(%arg1: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v"}, %arg2: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @"v2"})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK-DAG: "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}>
    %0 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>

    // CHECK-DAG: "tf.Const"() <{value = dense<2.000000e+00> : tensor<f32>}>
    %1 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }
}

// -----

// Test running the pass on a module that does not have
// tf_saved_model.semantics.
// CHECK-LABEL: module
module {}

// -----

!tf_res = tensor<!tf_type.resource<tensor<f32>>>
!tf_str = tensor<3x!tf_type.string>

// CHECK-LABEL: module attributes
module attributes {tf_saved_model.semantics} {

"tf_saved_model.global_tensor"() {sym_name = "v1", type = tensor<f32>, value = dense<3.0> : tensor<f32> } : () -> ()
"tf_saved_model.global_tensor"() {sym_name = "v2", type = tensor<f32>, value = dense<2.0> : tensor<f32> } : () -> ()

// CHECK-LABEL: @body
func.func private @body(%arg0: !tf_res, %arg1: !tf_res) -> (!tf_res, !tf_res) {
  %graph:2 = tf_executor.graph {
    %value, %value_control = tf_executor.island wraps "tf.GetKey"() : () -> tensor<f32>
    %ret0, %ret0_control = tf_executor.island wraps "tf.SomeOp"() : () -> !tf_res
    %ret1, %ret1_control = tf_executor.island wraps "tf.SomeOp"() : () -> !tf_res
    %control_unknown = tf_executor.island wraps "tf.UnknownOp"() : () -> ()
    %key, %key_control = tf_executor.island wraps "tf.GetKey"() : () -> !tf_str
    // CHECK: "tf.ReadVariableOp"(%arg0)
    %read1, %read1_control  = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (!tf_res) -> tensor<f32>
    // CHECK: "tf.ReadVariableOp"(%arg1)
    %read2, %read2_control  = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) : (!tf_res) -> tensor<f32>
    tf_executor.island wraps "tf.TPUExecuteAndUpdateVariables"(%ret0, %ret1, %key) {
        device_var_reads_indices = [0, 1],
        device_var_updates_indices = [0, 1]} : (!tf_res, !tf_res, !tf_str) -> ()
    tf_executor.fetch %ret0, %ret1: !tf_res, !tf_res
  }
  func.return %graph#0, %graph#1 : !tf_res, !tf_res
}

// CHECK-LABEL: @cond
func.func private @cond(%arg0: !tf_res, %arg1: !tf_res) -> (tensor<i32>) {
  %graph = tf_executor.graph {
    %island, %ctrl = tf_executor.island {
      %pred = "tf.SomeOp"() : () -> tensor<i32>
      tf_executor.yield %pred : tensor<i32>
    }
    tf_executor.fetch %island : tensor<i32>
  }
  func.return %graph : tensor<i32>
}

// CHECK-LABEL: @test_while_loop
func.func @test_while_loop(%arg0: !tf_res {tf._composite_device = "/job:tpu_host_worker/replica:0/task:0/device:COMPOSITE:0", tf_saved_model.bound_input = @v1},
                           %arg1: !tf_res {tf_saved_model.bound_input = @v2})
    attributes {tf_saved_model.exported_names = ["test_while_loop"]} {
  // CHECK-DAG: Const{{.*}}2.0
  // CHECK-DAG: Const{{.*}}3.0
  %read1 = "tf.ReadVariableOp"(%arg0) : (!tf_res) -> tensor<f32>
  %read2 = "tf.ReadVariableOp"(%arg1) : (!tf_res) -> tensor<f32>
  // CHECK: tf_executor.graph
  tf_executor.graph {
    %handle0, %handle0_control = tf_executor.island wraps "tf.SomeOp"() : () -> !tf_res
    %handle1, %handle1_control = tf_executor.island wraps "tf.SomeOp"() : () -> !tf_res
    %control_A = tf_executor.island wraps "tf.OpA"() : () -> ()
    %while_out:2, %while_control = tf_executor.island(%control_A) wraps "tf.While"(
            %handle0, %handle1) {
        body = @body, cond = @cond, is_stateless = false
    } : (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<!tf_type.resource<tensor<f32>>>, tensor<!tf_type.resource<tensor<f32>>>)
    %control_B = tf_executor.island(%while_control) wraps "tf.OpB"() : () -> ()
    tf_executor.fetch
  }
  func.return
}
}

// -----

// Test variable is frozen when it is used inside `TF::BatchFunctionOp` without
// assignment.

module attributes {tf_saved_model.semantics} {
  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "var1", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func private @f_batch_callee(%arg0: tensor<?xf32>, %arg1: tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<?xf32>, tensor<f32>) {
    %0 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return %arg0, %0 : tensor<?xf32>, tensor<f32>
  }
  // CHECK: func.func private @f_batch_callee(%[[ARG_0:.*]]: tensor<?xf32>) -> (tensor<?xf32>, tensor<f32>)
  // CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK: return %[[ARG_0]], %[[CST_0]] : tensor<?xf32>, tensor<f32>

  func.func @f(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> (tensor<*xf32> {tf_saved_model.index_path = []}, tensor<*xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %arg = "tf.Const"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %0, %1 = "tf.BatchFunction"(%arg, %handle) {f = @f_batch_callee, operandSegmentSizes = array<i32: 1, 1>, batch_timeout_micros = 1000, max_batch_size = 8, num_batch_threads = 2} : (tensor<1xf32>, tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<*xf32>, tensor<*xf32>)
    func.return %0, %1 : tensor<*xf32>, tensor<*xf32>
  }
  // CHECK: func.func @f() -> (tensor<*xf32> {tf_saved_model.index_path = []}, tensor<*xf32> {tf_saved_model.index_path = []}) attributes {tf_saved_model.exported_names = ["f"]} {
  // CHECK:  %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK:  %0:2 = "tf.BatchFunction"(%cst) <{batch_timeout_micros = 1000 : i64, f = @f_batch_callee, max_batch_size = 8 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 0>}> : (tensor<1xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  // CHECK:  return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
  // CHECK: }
}

// -----

// Tests that "tf._input_shapes" attribute is updated correctly

module attributes {tf_saved_model.semantics} {
  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "var1", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> (tensor<f32>  {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %cst = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    %val = "tf.PartitionedCall"(%cst, %handle) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<f32>, tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>)
    func.return %val : tensor<f32>
  }
  // CHECK: func.func @f() -> (tensor<f32> {tf_saved_model.index_path = []}) attributes {tf_saved_model.exported_names = ["f"]} {
  // CHECK:   %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:   %0 = "tf.PartitionedCall"(%cst) <{config = "", config_proto = "", executor_type = "", f = @f_callee}> : (tensor<f32>) -> tensor<f32>
  // CHECK:   return %0 : tensor<f32>
  // CHECK: }
  func.func private @f_callee(%arg0: tensor<f32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<f32> attributes {tf._input_shapes = [#tf_type.shape<0>, #tf_type.shape<>]} {
    %0 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> tensor<f32>
    %1 = "tf.AddV2"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    func.return %1 : tensor<f32>
  }
  // CHECK: func.func private @f_callee(%arg0: tensor<f32>) -> tensor<f32> attributes {tf._input_shapes = [#tf_type.shape<0>]} {
  // CHECK:   %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:   %0 = "tf.AddV2"(%arg0, %cst) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK:   return %0 : tensor<f32>
  // CHECK: }
}

// -----

// Test While region immutable case.

module attributes {tf_saved_model.semantics} {
  // CHECK-NOT: tf_saved_model.global_tensor
 "tf_saved_model.global_tensor"() {sym_name = "var1", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> (tensor<f32>  {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %res:2 = func.call @f_1(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>, tensor<f32>)
    func.return %res#0 : tensor<f32>
  }
  // CHECK: func.func @f() -> (tensor<f32> {tf_saved_model.index_path = []}) attributes {tf_saved_model.exported_names = ["f"]} {
  // CHECK:   %0:2 = call @f_1() : () -> (tensor<f32>, tensor<f32>)
  // CHECK:   return %0#0 : tensor<f32>
  // CHECK: }

  // CHECK: func private @f_1() -> (tensor<f32>, tensor<f32>)
  func.func private @f_1(%arg0: tensor<!tf_type.resource<tensor<f32>>>)-> (tensor<f32>, tensor<f32>) {
    %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %1:3 = "tf.WhileRegion"(%arg0, %0, %cst) ({
      ^bb0(%carg0: tensor<*x!tf_type.resource>, %carg1: tensor<i32>, %carg2 : tensor<f32>):
         %limit = arith.constant dense<5> : tensor<i32>
         %cond = "tf.Less"(%carg1, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
         "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },  {
      ^bb0(%barg0: tensor<*x!tf_type.resource>, %barg1: tensor<i32>, %barg2: tensor<f32>):
        %val = "tf.PartitionedCall"(%barg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<f32>)
        "tf.Yield"(%barg0, %barg1, %val) : (tensor<*x!tf_type.resource>,tensor<i32>, tensor<f32>) -> ()
    }) {is_stateless = true} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<i32>, tensor<f32>) -> (tensor<*x!tf_type.resource>, tensor<i32>, tensor<f32>)
    func.return %1#2, %1#2 : tensor<f32>, tensor<f32>
  }

  // CHECK: func.func private @f_callee_callee() -> tensor<f32> {
  func.func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<f32> {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource>) -> (tensor<f32>)
    func.return %0 : tensor<f32>
  }
  // CHECK:   %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:   return %cst : tensor<f32>
  // CHECK: }
}

// -----

// Make sure global tensors marked as immutable are not written to.
module attributes {tf_saved_model.semantics} {
 "tf_saved_model.global_tensor"() {sym_name = "var1", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> ()
  attributes {tf_saved_model.exported_names = ["f"]} {
    %cst = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
    // expected-error @+1 {{immutable bound input}}
    "tf.AssignVariableOp"(%handle, %cst) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    func.return
  }

  func.func @f2(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> (tensor<f32>  {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f2"]} {
    %0 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>)
    func.return %0 : tensor<f32>
  }
}

// -----

// Test If Region immutable case.

module attributes {tf_saved_model.semantics} {
 "tf_saved_model.global_tensor"() {sym_name = "var1", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%handle: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @var1}) -> (tensor<f32>  {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %arg0 = "tf.Const"() { value = dense<1> : tensor<i1> } : () -> tensor<i1>
    %0 = "tf.IfRegion"(%arg0) ({
      // CHECK-NOT: "tf.ReadVariableOp"
      %1 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>)
      "tf.Yield"(%1) : (tensor<f32>) -> ()
     },  {
      %2 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    func.return %0 : tensor<f32>
  }
}


