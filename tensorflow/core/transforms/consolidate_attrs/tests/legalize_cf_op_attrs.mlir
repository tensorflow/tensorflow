// RUN: tfg-transforms-opt --split-input-file --tfg-prepare-attrs-export %s | FileCheck %s

// CHECK-LABEL: tfg.func @test_if_attrs
tfg.func @test_if_attrs(%cond: tensor<i1>, %arg: tensor<i32>) -> (tensor<4xi32>) {
  // CHECK: If
  // CHECK-SAME: Tcond = i1, Tin = [i32], Tout = [i32]
  %If, %ctl = If(%cond, %arg) [%cond.ctl] {
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<i32>) -> (tensor<4xi32>)
  return(%If) : tensor<4xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_case_attrs
tfg.func @test_case_attrs(%branch: tensor<i32>, %arg: tensor<i32>) -> (tensor<2xi32>) {
  // CHECK: Case
  // CHECK-SAME: Tin = [i32], Tout = [i32]
  %Case, %ctl = Case(%branch, %arg) [%branch.ctl] {branches = []} : (tensor<i32>, tensor<i32>) -> (tensor<2xi32>)
  return(%Case) : tensor<2xi32>
}

// -----

// CHECK-LABEL: tfg.func @test_while_attrs
tfg.func @test_while_attrs(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: While
  // CHECK-SAME: T = [i32]
  %While, %ctl = While(%arg) {
    body = #tf_type.func<@body, {}>, cond = #tf_type.func<@cond, {}>,
    parallel_iterations = 10 : i64
  } : (tensor<i32>) -> (tensor<i32>)
  return(%While) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_for_attrs
tfg.func @test_for_attrs(%arg: tensor<i32>) -> (tensor<i32>) {
  // CHECK: For
  // CHECK: T = [i32]
  %For, %ctl = For(%arg, %arg, %arg, %arg) [%arg.ctl] {body = #tf_type.func<@body, {}>}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  return(%For) : tensor<i32>
}
