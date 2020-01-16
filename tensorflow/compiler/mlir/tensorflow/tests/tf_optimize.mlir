// RUN: tf-opt %s -tf-optimize | FileCheck %s

// CHECK-LABEL: @fuseMulIntoConv2d
func @fuseMulIntoConv2d(%arg0: tensor<1x112x112x3xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst2 = constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<1x112x112x3xf32>, tensor<1x3x3x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Mul"(%0, %cst2) : (tensor<1x112x112x2xf32>, tensor<2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %[[CST:.*]] = "tf.Const{{.*}} dense<
  // CHECK-SAME: [1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00], [5.000000e+00, 1.200000e+01]
  // CHECK-SAME: [7.000000e+00, 1.600000e+01], [9.000000e+00, 2.000000e+01], [1.100000e+01, 2.400000e+01]
  // CHECK-SAME: [1.300000e+01, 2.800000e+01], [1.500000e+01, 3.200000e+01], [1.700000e+01, 3.600000e+01]
  // CHECK: %[[CONV:.*]] = "tf.Conv2D"(%arg0, %[[CST]]) {data_format = "NHWC", dilations = [1, 2, 3, 1], explicit_paddings = [], padding = "SAME", strides = [1, 4, 5, 1], use_cudnn_on_gpu = true}
  // CHECK: return %[[CONV]] : tensor<1x112x112x2xf32>
}

// CHECK-LABEL: @notfuseMulIntoConv2d
// filter and multiply are not broadcastable
func @notfuseMulIntoConv2d(%arg0: tensor<1x112x112x3xf32>) -> tensor<1x112x112x2xf32> {
  %cst0 = constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst2 = constant dense<3.0> : tensor<112x2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<1x112x112x3xf32>, tensor<1x3x3x2xf32>) -> tensor<1x112x112x2xf32>
  %1 = "tf.Mul"(%0, %cst2) : (tensor<1x112x112x2xf32>, tensor<112x2xf32>) -> tensor<1x112x112x2xf32>

  return %1 : tensor<1x112x112x2xf32>
  // CHECK: %cst_0 = constant dense<3.000000e+00> : tensor<112x2xf32>
  // CHECK: %0 = "tf.Conv2D"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]}
  // CHECK: %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x112x112x2xf32>, tensor<112x2xf32>) -> tensor<1x112x112x2xf32>
  // CHECK: return %1 : tensor<1x112x112x2xf32>
}
