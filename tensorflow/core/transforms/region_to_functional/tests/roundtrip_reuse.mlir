// RUN: tfg-transforms-opt --tfg-functional-to-region --tfg-region-to-functional %s | FileCheck %s

// Check that roundtripping through region conversion will re-use the function
// if the bodies match.

// CHECK-LABEL: tfg.func @body
tfg.func @body(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  return(%arg0, %arg1) [%arg0.ctl] : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: tfg.func @cond
tfg.func @cond(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i1>) {
  %A, %ctl = A name("A") {_some_attr = 5 : i32} : () -> (tensor<i1>)
  return(%A) [%ctl] : tensor<i1>
}

// CHECK-LABEL: tfg.func @test
tfg.func @test(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: While
  // CHECK-SAME: body = #tf_type.func<@body, {a}>
  // CHECK-SAME: cond = #tf_type.func<@cond, {b}>
  %While:2, %ctl = While(%arg0, %arg1) name("while") {
    T = [i32, i32], output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
    body = #tf_type.func<@body, {a}>, cond = #tf_type.func<@cond, {b}>,
    parallel_iterations = 10 : i64
  } : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return(%While#0, %While#1) : tensor<i32>, tensor<i32>
}

// CHECK-NOT: tfg.func
