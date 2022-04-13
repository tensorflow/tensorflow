// RUN: tf-reduce %s -reduction-tree='traversal-mode=0 test=%S/reducer/unsupported-op-test.sh' | FileCheck %s

// CHECK: @target_function
func.func @target_function() -> tensor<i32> {
  %0 = "tf_device.cluster"() ({
    // CHECK: tf.UnsupportedOp
    %1 = "tf.UnsupportedOp"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: tf.Identity
    %2 = "tf.Identity"(%1) : (tensor<i32>) -> tensor<i32>
    // CHECK-NOT: tf.Identity
    %3 = "tf.Identity"(%2) : (tensor<i32>) -> tensor<i32>
    %4 = "tf.Identity"(%3) : (tensor<i32>) -> tensor<i32>
    // CHECK: tf_device.return
    tf_device.return %4 : tensor<i32>
  }) {num_cores_per_replica = 1, topology =  "", device_assignment =  []} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}
