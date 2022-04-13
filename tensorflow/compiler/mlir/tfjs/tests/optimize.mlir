// Run optimize pass only and check the results.
// RUN: tfjs-opt %s -tfjs-optimize | FileCheck %s

// CHECK-LABEL: prelu_fusion
func.func @prelu_fusion(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<-0.2> : tensor<3xf32>
  %0 = "tf.Relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tf.Neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tf.Relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tf.Mul"(%alpha, %2) : (tensor<3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tf.AddV2"(%0, %3) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = tfjs.Prelu
}

// CHECK-LABEL: prelu_not_fused
// Rank of alpha should be one less than input for PReLU, which is not the case.
func.func @prelu_not_fused(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %alpha = arith.constant dense<-0.2> : tensor<f32>
  %0 = "tf.Relu"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "tf.Neg"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "tf.Relu"(%1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = "tf.Mul"(%alpha, %2) : (tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %4 = "tf.AddV2"(%0, %3) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %4 : tensor<2x3xf32>

  // CHECK: %[[RESULT:[0-9].*]] = "tf.Relu"
}
