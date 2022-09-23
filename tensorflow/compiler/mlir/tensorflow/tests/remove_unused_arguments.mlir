// RUN: tf-opt %s -allow-unregistered-dialect --tf-remove-unused-arguments --split-input-file | FileCheck %s

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  "SomeOp"(%arg1) : (tensor<f32>) -> ()
  return %arg1 : tensor<f32>
}

// CHECK-LABEL: removes_first_arg
func.func @removes_first_arg(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: call{{.*}}(%arg1)
  %1 = func.call @f(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  "SomeOp"(%arg0) : (tensor<f32>) -> ()
  return %arg0 : tensor<f32>
}

// CHECK-LABEL: removes_last_arg
func.func @removes_last_arg(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: call{{.*}}(%arg0)
  %1 = func.call @f(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  "SomeOp"(%arg1) : (tensor<f32>) -> ()
  return %arg1 : tensor<f32>
}

// CHECK-LABEL: leaves_public_functions_alone
func.func @leaves_public_functions_alone(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: call{{.*}}(%arg0, %arg1)
  %1 = func.call @f(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  return
}

// CHECK-LABEL: removes_all_args
func.func @removes_all_args(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  // CHECK: call{{.*}}() :
  func.call @f(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> ()
  return
}

// -----

// CHECK-LABEL: handles_mlprogram
// CHECK-SAME: () {
ml_program.func private @handles_mlprogram(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  ml_program.return
}

// -----

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// CHECK-LABEL: handles_partitioned_function_calls
func.func @handles_partitioned_function_calls(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: PartitionedCall"()
  %1 = "tf.PartitionedCall"(%arg0, %arg1) {f = @f} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----

func.func private @f(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> f32 {
    %0 = "tf.Add2"(%arg0, %arg2) : (f32, f32) -> f32
    return %0 : f32
}

// CHECK-LABEL: removes_multiple_args
func.func @removes_multiple_args(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK: call{{.*}}(%arg0, %arg0)
  %0 = func.call @f(%arg0, %arg1, %arg0, %arg1) : (f32, f32, f32, f32) -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: leaves_while_loops_alone
func.func @leaves_while_loops_alone(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  // CHECK: While{{.*}}(%arg0, %arg1)
  %0, %1 = "tf.While"(%arg0, %arg1) {body = @body, cond = @cond, is_stateless = false} :
      (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  func.return
}

// CHECK: @body(%arg0{{.*}}, %arg1{{.*}})
func.func private @body(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %0 = "tf.Const"() { value = dense<42> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Const"() { value = dense<42.0> : tensor<f32> } : () -> tensor<f32>
  func.return %0, %1 : tensor<i32>, tensor<f32>
}

// CHECK: @cond(%arg0{{.*}}, %arg1{{.*}})
func.func private @cond(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<i1> {
  %const = "tf.Const"() { value = dense<1000> : tensor<i32> } : () -> tensor<i32>
  %result = "tf.Less"(%const, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %result : tensor<i1>
}

// -----

func.func private @f(%arg0: f32, %arg1: f32) -> (f32, f32) {
  %1 = "some_op"(%arg1) : (f32) -> f32
  return %arg0, %1 : f32, f32
}

// CHECK-LABEL: @removes_first_passthrough_arg
func.func @removes_first_passthrough_arg(%arg0: f32, %arg1: f32) -> (f32, f32) {
  // CHECK: %0 = call @f(%arg1)
  %0, %1 = call @f(%arg0, %arg1) : (f32, f32) -> (f32, f32)
  // CHECK: return %arg0, %0
  return %0, %1 : f32, f32
}

// -----

func.func private @f(%arg0: f32, %arg1: f32) -> (f32, f32) {
  %0 = "some_op"(%arg0) : (f32) -> f32
  return %0, %arg1 : f32, f32
}

// CHECK-LABEL: @removes_second_passthrough_arg
func.func @removes_second_passthrough_arg(%arg0: f32, %arg1: f32) -> (f32, f32) {
  // CHECK: %0 = call @f(%arg0)
  %0, %1 = call @f(%arg0, %arg1) : (f32, f32) -> (f32, f32)
  // CHECK: return %0, %arg1
  return %0, %1 : f32, f32
}

// -----

func.func private @f(%arg0: f32) -> (f32, f32) {
  return %arg0, %arg0 : f32, f32
}

// CHECK-LABEL: @can_remove_all_results
func.func @can_remove_all_results(%arg0: f32) -> (f32, f32) {
  // CHECK: call @f()
  %0, %1 = call @f(%arg0) : (f32) -> (f32, f32)
  // CHECK: return %arg0, %arg0
  return %0, %1 : f32, f32
}

// -----

// CHECK-LABEL: @has_inner_function
func.func private @has_inner_function(%arg0: f32) -> (f32, f32) {
  func.func private @inner() -> (tensor<f32>, tensor<f32>) {
    %0, %1 = "some_constant"() : () -> (tensor<f32>, tensor<f32>)
    // CHECK: return
    // CHECK-SAME: tensor<f32>, tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
  // CHECK: return
  // CHECK-NOT: arg
  return %arg0, %arg0 : f32, f32
}

// CHECK-LABEL: @respects_regions
func.func @respects_regions(%arg0: f32) -> (f32, f32) {
  // CHECK: call @has_inner_function()
  %0, %1 = call @has_inner_function(%arg0) : (f32) -> (f32, f32)
  // CHECK: return %arg0, %arg0
  return %0, %1 : f32, f32
}

// -----

// CHECK-LABEL: @handles_recursion
// CHECK-SAME: %arg0: f32
func.func private @handles_recursion(%arg0: f32, %arg1: f32) -> (f32, f32) {
  // CHECK: call @handles_recursion(%arg0)
  %0, %1 = call @handles_recursion(%arg0, %arg0) : (f32, f32) -> (f32, f32)
  // CHECK: return
  // CHECK-SAME: f32, f32
  return %0, %1 : f32, f32
}

// -----

// CHECK-LABEL: @handles_multiple_returns
// CHECK-SAME: %arg0
// CHECK-SAME: %arg1
func.func private @handles_multiple_returns(%arg0: f32, %arg1: f32) -> (f32, f32) {
  "SomeOp"() : () -> ()
 ^bb0:
  return %arg0, %arg1 : f32, f32
 ^bb1:
  return %arg1, %arg0 : f32, f32
}

// -----

// CHECK-LABEL: @removes_multiple_returns
func.func private @removes_multiple_returns(%arg0: f32, %arg1: f32) -> (f32, f32) {
  "SomeOp"() : () -> ()
 ^bb0:
  // CHECK: return
  // CHECK-NOT: arg
  return %arg0, %arg1 : f32, f32
 ^bb1:
  // CHECK: return
  // CHECK-NOT: arg
  return %arg0, %arg1 : f32, f32
}
