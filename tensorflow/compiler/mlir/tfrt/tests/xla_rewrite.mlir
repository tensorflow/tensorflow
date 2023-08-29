// RUN: tf-tfrt-opt -tfrt-xla-rewrite %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: xla_launch
func.func @xla_launch(%arg: tensor<i32>, %v0: tensor<*x!tf_type.resource>, %v1: tensor<*x!tf_type.resource>) -> (tensor<i32>) {
  %c0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %c1 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // CHECK: tf.XlaLaunchV2
  // CHECK-SAME: constants = [0, 3]
  // CHECK-SAME: resources = [2, 4]
  %r0 = "tf.StatefulPartitionedCall"(%c0, %arg, %v0, %c1, %v1)
    {_XlaMustCompile = true, config = "", config_proto = "",
      device = "/device:CPU:0", executor_type = "", f = @callee}
      : (tensor<i32>, tensor<i32>, tensor<*x!tf_type.resource>, tensor<i32>, tensor<*x!tf_type.resource>) -> tensor<i32>

  // CHECK-NOT: tf.XlaLaunchV2
  // CHECK: tf.StatefulPartitionedCall
  %r1 = "tf.StatefulPartitionedCall"(%c0, %r0, %v0, %c1, %v1)
    {_XlaMustCompile = false, config = "", config_proto = "",
      device = "/device:CPU:0", executor_type = "", f = @callee}
      : (tensor<i32>, tensor<i32>, tensor<*x!tf_type.resource>, tensor<i32>, tensor<*x!tf_type.resource>) -> tensor<i32>

  // CHECK-NOT: tf.XlaLaunchV2
  // CHECK: tf.StatefulPartitionedCall
  %r2 = "tf.StatefulPartitionedCall"(%c0, %r1, %v0, %c1, %v1)
    {_XlaMustCompile = true, config = "", config_proto = "",
      device = "/device:GPU:0", executor_type = "", f = @callee}
      : (tensor<i32>, tensor<i32>, tensor<*x!tf_type.resource>, tensor<i32>, tensor<*x!tf_type.resource>) -> tensor<i32>

  func.return %r2 : tensor<i32>
}

func.func @callee(%c0: tensor<i32>, %arg: tensor<i32>, %v0: tensor<*x!tf_type.resource>, %c1: tensor<i32>, %v1: tensor<*x!tf_type.resource>) -> (tensor<i32>) {
    %0 = "tf.ReadVariableOp"(%v0) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<i32>
    %1 = "tf.ReadVariableOp"(%v1) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<i32>
    %r = "tf.AddN"(%c0, %c1, %arg, %0, %1) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    func.return %r : tensor<i32>
}
