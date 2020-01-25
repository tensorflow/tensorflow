// RUN: tf-opt %s -tf-device-constant-sinking | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @sink_const
func @sink_const(%arg0 : tensor<16xf32>) -> (tensor<16xf32>, tensor<f32>) {
  // Verify that the constant are sunk in the tf_device.launch region using them
  // and removed if no other use is left.

  // Only the 2.0 and 3.0 constants are removed, the 4.0 has a use in the return
  // CHECK-NOT:"tf.Const"2.0
  // CHECK-NOT:"tf.Const"3.0
  %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %2 = "tf.Const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %3 = tf_executor.graph {
    %res, %ctl = tf_executor.island {
      %3 = "tf_device.launch"() ({

        // In the device region, check that the 3 constants are materialized and
        // remapped to the uses.
        // CHECK: tf_device.launch
        // CHECK-DAG: %[[CST2:.*]] = "tf.Const"{{.*}}2.0
        // CHECK-DAG: %[[CST3:.*]] = "tf.Const"{{.*}}3.0
        // CHECK-DAG: %[[CST4:.*]] = "tf.Const"{{.*}}4.0
        // CHECK-NOT:"tf.Const"
        // CHECK: %[[MUL1:.*]] = "tf.Mul"(%arg0, %[[CST2]])
        // CHECK-NEXT: %[[MUL2:.*]] = "tf.Mul"(%[[MUL1]], %[[CST2]])
        // CHECK-NEXT: %[[MUL3:.*]] = "tf.Mul"(%[[MUL2]], %[[CST3]])
        // CHECK-NEXT: = "tf.Mul"(%[[MUL3]], %[[CST4]])
        %3 = "tf.Mul"(%arg0, %0) : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
        %4 = "tf.Mul"(%3, %0) : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
        %5 = "tf.Mul"(%4, %1) : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
        %6 = "tf.Mul"(%5, %2) : (tensor<16xf32>, tensor<f32>) -> tensor<16xf32>
        tf_device.return %6 : tensor<16xf32>
      }) {device = "tpu0"} : () -> tensor<16xf32>
      tf_executor.yield %3 : tensor<16xf32>
    }
    tf_executor.fetch %res : tensor<16xf32>
  }
  return %3, %2 : tensor<16xf32>, tensor<f32>
}

