// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs --split-input-file | FileCheck %s

// CHECK-LABEL: tfg.func @test_drop_if_attrs
tfg.func @test_drop_if_attrs(%arg0: tensor<i1>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: If
  // CHECK-NOT: Tcond
  // CHECK-NOT: Tin
  // CHECK-NOT: Tout
  %If, %ctl = If(%arg0, %arg1) {
    Tcond = i1, Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (tensor<i1>, tensor<i32>) -> (tensor<i32>)
  // CHECK: return
  return(%If) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_drop_case_attrs
tfg.func @test_drop_case_attrs(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: Case
  // CHECK-NOT: Tin
  // CHECK-NOT: Tout
  %Case, %ctl = Case(%arg0, %arg1) {
    Tin = [i32], Tout = [i32], output_shapes = [#tf_type.shape<>], branches = []
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: return
  return(%Case) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_drop_while_attrs
tfg.func @test_drop_while_attrs(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: While
  // CHECK-NOT: T
  %While, %ctl = While(%arg0) {
    T = [i32], output_shapes = [#tf_type.shape<>], parallel_iterations = 10 : i64,
    cond = #tf_type.func<@cond, {}>, body = #tf_type.func<@body, {}>
  } : (tensor<i32>) -> (tensor<i32>)
  // CHECK: return
  return(%While) : tensor<i32>
}

// -----

// CHECK-LABEL: tfg.func @test_drop_for_attrs
tfg.func @test_drop_for_attrs(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: For
  // CHECK-NOT: T
  %For, %ctl = For(%arg0, %arg0, %arg0, %arg0) {
    T = [i32], body = #tf_type.func<@body, {}>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>)
  // CHECK: return
  return(%For) : tensor<i32>
}
