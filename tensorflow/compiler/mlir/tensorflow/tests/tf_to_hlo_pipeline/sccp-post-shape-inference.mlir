// RUN: tf-opt -tf-to-hlo-pipeline %s | FileCheck %s

// Verifies that constants generated post shape inference are propagated.
// get_shape result in this test.
module attributes {tf.versions = {producer = 179 : i32}} {

  // CHECK-LABEL: func @main
  func @main(%arg0: tensor<10x19xf32>, %arg1: tensor<19x10xf32> {mhlo.is_same_data_across_replicas}) -> tensor<?xi64> {
    %0 = "tf.Shape"(%arg0) : (tensor<10x19xf32>) -> tensor<2xi64>
    %1 = "tf.Reshape"(%arg1, %0) : (tensor<19x10xf32>, tensor<2xi64>) -> tensor<?x?xf32>

    // CHECK: %[[RESULT:.*]] = mhlo.constant dense<[10, 19]>
    %2 = "tf.PartitionedCall"(%1) {config = "", config_proto = "", executor_type = "", f = @get_shape} : (tensor<?x?xf32>) -> (tensor<?xi64>)

    // CHECK: return %[[RESULT]]
    return %2 : tensor<?xi64>
  }

  // CHECK-LABEL: func @get_shape
  func @get_shape(%arg0 : tensor<*xi64>) -> tensor<?xi64> {
    %0 = "tf.Shape"(%arg0) : (tensor<*xi64>) -> tensor<?xi64>
    return %0 : tensor<?xi64>
  }

}

