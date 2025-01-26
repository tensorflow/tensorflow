// RUN: tf-opt %s -tf-replicate-invariant-op-hoisting -tf-dialect-to-executor-v2 | FileCheck %s

// CHECK-LABEL: @concat_that_uses_replicated_args
// CHECK: Concat
// CHECK-NOT: Concat
func.func @concat_that_uses_replicated_args(
        %arg0: tensor<128x10xf32>,
        %arg1: tensor<128x10xf32>,
        %arg2: tensor<*xi32>,
        %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) {
  %0:4 = tf_device.replicate([%arg0, %arg1] as %arg4: tensor<128x10xf32>,
                             [%arg2, %arg3] as %arg5: tensor<*xi32>) {n = 2 : i32} {
    %6 = "tf.Something"(%arg4) : (tensor<128x10xf32>) -> tensor<*xi32>
    %cst_0 = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %5 = "tf.Concat"(%cst_0, %arg5, %6) : (tensor<i32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    tf_device.return %5, %6 : tensor<*xi32>, tensor<*xi32>
  }
  return %0#0, %0#0, %0#2, %0#3 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: @concat_that_does_not_use_replicated_args
// CHECK: Concat
// CHECK-NOT: Concat
func.func @concat_that_does_not_use_replicated_args(
        %arg0: tensor<128x10xf32>,
        %arg1: tensor<128x10xf32>,
        %arg2: tensor<*xi32>,
        %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) {
  %0:4 = tf_device.replicate([%arg0, %arg1] as %arg4: tensor<128x10xf32>,
                             [%arg2, %arg3] as %arg5: tensor<*xi32>) {n = 2 : i32} {
    %6 = "tf.Something"(%arg4) : (tensor<128x10xf32>) -> tensor<*xi32>
    %cst_0 = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %5 = "tf.Concat"(%cst_0, %arg2, %arg3) : (tensor<i32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    tf_device.return %5, %6 : tensor<*xi32>, tensor<*xi32>
  }
  return %0#0, %0#1, %0#2, %0#3 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>
}
