// RUN: tf-opt %s -allow-unregistered-dialect --tf-remove-unused-arguments --split-input-file | FileCheck %s

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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

func.func private @f(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  return
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
