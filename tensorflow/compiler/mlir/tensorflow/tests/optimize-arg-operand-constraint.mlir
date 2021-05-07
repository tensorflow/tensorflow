// RUN: tf-opt -tf-optimize %s | FileCheck %s

// Check passing an argument into DefinedByConv2D constraint does not crash.

// CHECK-LABEL: func @main
func @main(%arg0: tensor<1xf32>) -> tensor<1xf32>
attributes  {tf.entry_function = {inputs = "input", outputs = "output_node"}} {
  %0 = constant dense<2.000000e+00> : tensor<f32>
  %1 = constant dense<1.000000e+00> : tensor<f32>
  %2 = "tf.AddV2"(%arg0, %1) {T = "tfdtype$DT_FLOAT", device = "", name = "StatefulPartitionedCall/add"} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %3 = "tf.Mul"(%2, %0) {T = "tfdtype$DT_FLOAT", device = "", name = "output_node"} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  return %3 : tensor<1xf32>
}
