// RUN: dtensor-opt %s -split-input-file -dtensor-function-renaming -verify-diagnostics | FileCheck %s

module attributes {dtensor.cache_key = "_abc_def"}  {
  // CHECK-LABEL: func @main
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    // CHECK:       "tf.StatefulPartitionedCall"
    // CHECK-SAME:  f = @_func_0_abc_def
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1) {f = @_func_0, config = "", config_proto = "", executor_type = ""} : (tensor<f32>, tensor<4xf32>) -> (tensor<4xf32>)
    func.return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func private @_func_0_abc_def
  func.func private @_func_0(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
    %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<f32>, tensor<4xf32>) -> (tensor<4xf32>)
    func.return %0 : tensor<4xf32>
  }
}
