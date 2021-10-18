// RUN: tf-opt %s --tf-guarantee-all-funcs-one-use --tf-shape-inference | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-LABEL: func @while_main
// CHECK: %0 = "tf.While"(%arg0)
// CHECK-SAME: body = @while_body
// CHECK-SAME: cond = @while_cond
// CHECK: "tf.While"(%arg1)
// CHECK-SAME: body = @while_body_0
// CHECK-SAME: cond = @while_cond_1

// CHECK: func @while_body(%arg0: tensor<256x256xi32>)
// CHECK: func @while_cond(%arg0: tensor<256x256xi32>)
// CHECK: func private @while_body_0(%arg0: tensor<128xi32>)
// CHECK: func private @while_cond_1(%arg0: tensor<128xi32>)
func @while_main(%arg0: tensor<256x256xi32>, %arg1: tensor<128xi32>) -> (tensor<256x256xi32>, tensor<128xi32>) {
  %0 = "tf.While"(%arg0) {body = @while_body, cond = @while_cond, device = "", is_stateless = true} : (tensor<256x256xi32>) -> (tensor<256x256xi32>)
  %1 = "tf.While"(%arg1) {body = @while_body, cond = @while_cond, device = "", is_stateless = true} : (tensor<128xi32>) -> (tensor<128xi32>)
  return %0, %1: tensor<256x256xi32>, tensor<128xi32>
}

func @while_body(%arg0: tensor<*xi32>) -> (tensor<*xi32>) {
  %0 = "tf.Rank"(%arg0) : (tensor<*xi32>) -> tensor<i32>
  %1 = "tf.Add"(%0, %arg0): (tensor<i32>, tensor<*xi32>) -> tensor<*xi32>
  return %1: tensor<*xi32>
}

func @while_cond(%arg0: tensor<*xi32>) -> tensor<*xi1> {
  %cst = arith.constant dense<10> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {T = i32, device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}
}
