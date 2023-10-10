// RUN: tfg-transforms-opt --split-input-file --tfg-prepare-attrs-export %s | FileCheck %s

// CHECK-LABEL: tfg.func @test_if_output_shapes
tfg.func @test_if_output_shapes(%cond: tensor<i1>, %arg: tensor<i32>) -> (tensor<4xi32>) {
  // CHECK: If
  // CHECK-SAME: output_shapes = [#tf_type.shape<4>]
  %If, %ctl = If(%cond, %arg) {
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<i32>) -> (tensor<4xi32>)
  return(%If) : tensor<4xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_case_output_shapes
tfg.func @test_case_output_shapes(%branch: tensor<i32>, %arg: tensor<i32>) -> (tensor<2xi32>) {
  // CHECK: Case
  // CHECK-SAME: output_shapes = [#tf_type.shape<2>]
  %Case, %ctl = Case(%branch, %arg) {branches = []} : (tensor<i32>, tensor<i32>) -> (tensor<2xi32>)
  return(%Case) : tensor<2xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_while_output_shapes
tfg.func @test_while_output_shapes(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: While
  // CHECK-SAME: output_shapes = [#tf_type.shape<>]
  %While, %ctl = While(%arg) {
    body = #tf_type.func<@body, {}>, cond = #tf_type.func<@cond, {}>,
    parallel_iterations = 10 : i64
  } : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}
