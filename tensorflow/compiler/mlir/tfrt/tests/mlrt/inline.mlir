// RUN: tf-tfrt-opt -split-input-file -pass-pipeline='builtin.module(tf-to-mlrt, inline)' %s | FileCheck %s -dump-input=fail

// Test generated tf_mlrt while body and predicate is inlined.

func.func @then(%x: tensor<i1>, %y: tensor<i1>, %z: tensor<i32>) -> tensor<i1> {
  return %x: tensor<i1>
}

func.func @else(%x: tensor<i1>, %y: tensor<i1>, %z: tensor<i32>) -> tensor<i1> {
  return %y: tensor<i1>
}

// CHECK-LABEL: func @while_cond_if
// CHECK: [[cond:%.*]] = tf_mlrt.predicate
// CHECK: [[z:%.*]] = mlrt.cond [[cond]] @then @else
// CHECK: return [[z]]
func.func @while_cond_if(%cond: tensor<i1>, %x: tensor<i1>, %y: tensor<i1>, %z: tensor<i32>) -> (tensor<i1>) {
  %r = "tf.If"(%cond, %x, %y, %z) {then_branch = @then, else_branch = @else, is_stateless = true} : (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>) -> tensor<i1>
  return %r : tensor<i1>
}

// CHECK-LABEL: func @while_body_if
func.func @while_body_if(%cond: tensor<i1>, %x: tensor<i1>, %y: tensor<i1>, %z: tensor<i32>) -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>) {
  %0 = "tf.Const"() {__op_key = 0: i32, device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%z, %0) {__op_key = 1: i32, device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %cond, %x, %y, %1 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>
}

// CHECK-LABEL: func @while_test_if
// CHECK-SAME: -> !tf_mlrt.tensor
func.func @while_test_if(%cond: tensor<i1>, %x: tensor<i1>, %y: tensor<i1>) -> (tensor<i32>) {
  // CHECK: [[CONST:%.*]] = tf_mlrt.constop {tensor_proto = "\08\03\12\00"}
  %cst = "tf.Const"() {__op_key = 2: i32, device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // Predicate should be inlined.
  // CHECK-NEXT: tf_mlrt.predicate
  // CHECK-NEXT: mlrt.cond
  // CHECK-NEXT: tf_mlrt.predicate

  // CHECK-NEXT: mlrt.while
  %0:4 = "tf.While"(%cond, %x, %y, %cst) { cond = @while_cond_if, body = @while_body_if, is_stateless = false, parallel_iterations = 1} : (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>) -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>)
  // CHECK: return
  // CHECK-SAME: !tf_mlrt.tensor
  func.return %0#3 : tensor<i32>
}

// CHECK-LABEL: func @"while_body_if/tf_mlrt_body"
// CHECK-NOT: call

// CHECK-LABEL: func @"while_cond_if/tf_mlrt_predicate"
// CHECK-NOT: call
