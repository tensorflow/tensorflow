// RUN: tf-opt %s -tf-optimize | FileCheck %s

// CHECK-LABEL: @fuseMulIntoConv2d
func @fuseMulIntoConv2d(%arg0: tensor<1x112x112x3xf32>) -> tensor<1x28x23x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst2 = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<1x112x112x3xf32>, tensor<1x3x3x2xf32>) -> tensor<1x28x23x2xf32>
  %1 = "tf.Mul"(%0, %cst2) : (tensor<1x28x23x2xf32>, tensor<2xf32>) -> tensor<1x28x23x2xf32>

  return %1 : tensor<1x28x23x2xf32>
  // CHECK: %[[CST:.*]] = "tf.Const{{.*}} dense<
  // CHECK-SAME: [1.000000e+00, 4.000000e+00], [3.000000e+00, 8.000000e+00], [5.000000e+00, 1.200000e+01]
  // CHECK-SAME: [7.000000e+00, 1.600000e+01], [9.000000e+00, 2.000000e+01], [1.100000e+01, 2.400000e+01]
  // CHECK-SAME: [1.300000e+01, 2.800000e+01], [1.500000e+01, 3.200000e+01], [1.700000e+01, 3.600000e+01]
  // CHECK: %[[CONV:.*]] = "tf.Conv2D"(%arg0, %[[CST]]) {data_format = "NHWC", dilations = [1, 2, 3, 1], explicit_paddings = [], padding = "SAME", strides = [1, 4, 5, 1], use_cudnn_on_gpu = true}
  // CHECK: return %[[CONV]] : tensor<1x28x23x2xf32>
}

// CHECK-LABEL: @notfuseMulIntoConv2d
// filter and multiply are not broadcastable
func @notfuseMulIntoConv2d(%arg0: tensor<1x112x112x3xf32>) -> tensor<1x28x23x2xf32> {
  %cst0 = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]]]> : tensor<1x3x3x2xf32>
  %cst2 = arith.constant dense<3.0> : tensor<23x2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]} : (tensor<1x112x112x3xf32>, tensor<1x3x3x2xf32>) -> tensor<1x28x23x2xf32>
  %1 = "tf.Mul"(%0, %cst2) : (tensor<1x28x23x2xf32>, tensor<23x2xf32>) -> tensor<1x28x23x2xf32>

  return %1 : tensor<1x28x23x2xf32>
  // CHECK: %cst_0 = arith.constant dense<3.000000e+00> : tensor<23x2xf32>
  // CHECK: %0 = "tf.Conv2D"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 2, 3, 1], padding = "SAME", strides = [1, 4, 5, 1]}
  // CHECK: %1 = "tf.Mul"(%0, %cst_0) : (tensor<1x28x23x2xf32>, tensor<23x2xf32>) -> tensor<1x28x23x2xf32>
  // CHECK: return %1 : tensor<1x28x23x2xf32>
}


// CHECK-LABEL: simplifyBroadcastReshape
func @simplifyBroadcastReshape(%arg0: tensor<1x8x1x1x1x1x1x18xbf16>) -> tensor<8x6x6x18xbf16> {
  %cst_1 = arith.constant dense<[1, 8, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x8x6x1x6x1x1x18xbf16>
  %cst_2 = arith.constant dense<[8, 6, 6, 18]> : tensor<4xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<1x8x6x1x6x1x1x18xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  return %98 : tensor<8x6x6x18xbf16>

  // CHECK-DAG: %[[CST:.*]] = "tf.Const"() {value = dense<[8, 1, 1, 18]> : tensor<4xi64>} : () -> tensor<4xi64>
  // CHECK-DAG: %[[CST1:.*]] =  "tf.Const"() {value = dense<[8, 6, 6, 18]> : tensor<4xi64>} : () -> tensor<4xi64>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[CST]]) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<4xi64>) -> tensor<8x1x1x18xbf16>
  // CHECK: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[RESHAPE]], %[[CST1]]) : (tensor<8x1x1x18xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  // CHECK: return %[[BROADCAST]] : tensor<8x6x6x18xbf16>
}

// CHECK-LABEL: simplifyBroadcastReshapeExtraDims
func @simplifyBroadcastReshapeExtraDims(%arg0: tensor<1x8x1x1x1x1x1x18xbf16>) -> tensor<7x8x6x6x18xbf16> {
  %cst_1 = arith.constant dense<[7, 1, 8, 6, 1, 6, 1, 1, 18]> : tensor<9xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<9xi64>) -> tensor<7x1x8x6x1x6x1x1x18xbf16>
  %cst_2 = arith.constant dense<[7, 8, 6, 6, 18]> : tensor<5xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<7x1x8x6x1x6x1x1x18xbf16>, tensor<5xi64>) -> tensor<7x8x6x6x18xbf16>
  return %98 : tensor<7x8x6x6x18xbf16>

  // CHECK-DAG: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 8, 1, 1, 18]> : tensor<5xi64>} : () -> tensor<5xi64>
  // CHECK-DAG: %[[CST1:.*]] =  "tf.Const"() {value = dense<[7, 8, 6, 6, 18]> : tensor<5xi64>} : () -> tensor<5xi64>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[CST]]) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<5xi64>) -> tensor<1x8x1x1x18xbf16>
  // CHECK: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[RESHAPE]], %[[CST1]]) : (tensor<1x8x1x1x18xbf16>, tensor<5xi64>) -> tensor<7x8x6x6x18xbf16>
  // CHECK: return %[[BROADCAST]] : tensor<7x8x6x6x18xbf16>
}

// CHECK-LABEL: simplifyBroadcastReshapeOnes
func @simplifyBroadcastReshapeOnes(%arg0: tensor<1x1x1x1x1x1x1x18xbf16>) -> tensor<1x6x1x6x18xbf16> {
  %cst_1 = arith.constant dense<[1, 1, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x1x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x1x6x1x6x1x1x18xbf16>
  %cst_2 = arith.constant dense<[1, 6, 1, 6, 18]> : tensor<5xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<1x1x6x1x6x1x1x18xbf16>, tensor<5xi64>) -> tensor<1x6x1x6x18xbf16>
  return %98 : tensor<1x6x1x6x18xbf16>

  // CHECK-DAG: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 1, 1, 1, 18]> : tensor<5xi64>} : () -> tensor<5xi64>
  // CHECK-DAG: %[[CST1:.*]] = "tf.Const"() {value = dense<[1, 6, 1, 6, 18]> : tensor<5xi64>} : () -> tensor<5xi64>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[CST]]) : (tensor<1x1x1x1x1x1x1x18xbf16>, tensor<5xi64>) -> tensor<1x1x1x1x18xbf16>
  // CHECK: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%[[RESHAPE]], %[[CST1]]) : (tensor<1x1x1x1x18xbf16>, tensor<5xi64>) -> tensor<1x6x1x6x18xbf16>
  // CHECK: return %[[BROADCAST]] : tensor<1x6x1x6x18xbf16>
}

// CHECK-LABEL: avoidSimplifyBroadcastReshape
func @avoidSimplifyBroadcastReshape(%arg0: tensor<1x8x1x1x1x1x1x18xbf16>) -> (tensor<1x8x6x1x6x1x1x18xbf16>, tensor<8x6x6x18xbf16>) {
  %cst_1 = arith.constant dense<[1, 8, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x8x6x1x6x1x1x18xbf16>
  %cst_2 = arith.constant dense<[8, 6, 6, 18]> : tensor<4xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<1x8x6x1x6x1x1x18xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  return %97, %98 : tensor<1x8x6x1x6x1x1x18xbf16>, tensor<8x6x6x18xbf16>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1, 8, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<[8, 6, 6, 18]> : tensor<4xi64>
  // CHECK: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%arg0, %[[CST]]) : (tensor<1x8x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x8x6x1x6x1x1x18xbf16>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%[[BROADCAST]], %[[CST1]]) : (tensor<1x8x6x1x6x1x1x18xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  // CHECK: return %[[BROADCAST]], %[[RESHAPE]] : tensor<1x8x6x1x6x1x1x18xbf16>, tensor<8x6x6x18xbf16>
}

// CHECK-LABEL: avoidSimplifyBroadcastReshapeUnmatchedDims
// The reshape splits broadcasted dimensions, instead of eliminating size-1 dimensions.
// This results in a mismatch between the non-unit dimensions in the input and output.
func @avoidSimplifyBroadcastReshapeUnmatchedDims(%arg0: tensor<1x1x1x1x1x1x1x18xbf16>) -> tensor<1x3x2x1x3x2x18xbf16> {
  %cst_1 = arith.constant dense<[1, 1, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x1x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x1x6x1x6x1x1x18xbf16>
  %cst_2 = arith.constant dense<[1, 3, 2, 1, 3, 2, 18]> : tensor<7xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<1x1x6x1x6x1x1x18xbf16>, tensor<7xi64>) -> tensor<1x3x2x1x3x2x18xbf16>
  return %98 : tensor<1x3x2x1x3x2x18xbf16>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<[1, 1, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  // CHECK-DAG: %[[CST1:.*]] = arith.constant dense<[1, 3, 2, 1, 3, 2, 18]> : tensor<7xi64>
  // CHECK: %[[BROADCAST:.*]] = "tf.BroadcastTo"(%arg0, %[[CST]]) : (tensor<1x1x1x1x1x1x1x18xbf16>, tensor<8xi64>) -> tensor<1x1x6x1x6x1x1x18xbf16>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%[[BROADCAST]], %[[CST1]]) : (tensor<1x1x6x1x6x1x1x18xbf16>, tensor<7xi64>) -> tensor<1x3x2x1x3x2x18xbf16>
  // CHECK: return %[[RESHAPE]] : tensor<1x3x2x1x3x2x18xbf16>
}

// CHECK-LABEL: avoidSimplifyBroadcastReshapeUnknownDims
func @avoidSimplifyBroadcastReshapeUnknownDims(%arg0: tensor<1x?x1x1x1x1x1x?xbf16>) -> tensor<8x6x6x18xbf16> {
  %cst_1 = arith.constant dense<[1, -1, 6, 1, 6, 1, 1, -1]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<1x?x1x1x1x1x1x?xbf16>, tensor<8xi64>) -> tensor<1x?x6x1x6x1x1x?xbf16>
  %cst_2 = arith.constant dense<[8, 6, 6, 18]> : tensor<4xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<1x?x6x1x6x1x1x?xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  return %98 : tensor<8x6x6x18xbf16>

  // CHECK: "tf.BroadcastTo"
  // CHECK: "tf.Reshape"
}

// CHECK-LABEL: avoidSimplifyBroadcastReshapeUnknownRanks
func @avoidSimplifyBroadcastReshapeUnknownRanks(%arg0: tensor<*xbf16>) -> tensor<8x6x6x18xbf16> {
  %cst_1 = arith.constant dense<[1, 8, 6, 1, 6, 1, 1, 18]> : tensor<8xi64>
  %97 = "tf.BroadcastTo"(%arg0, %cst_1) : (tensor<*xbf16>, tensor<8xi64>) -> tensor<*xbf16>
  %cst_2 = arith.constant dense<[8, 6, 6, 18]> : tensor<4xi64>
  %98 = "tf.Reshape"(%97, %cst_2) : (tensor<*xbf16>, tensor<4xi64>) -> tensor<8x6x6x18xbf16>
  return %98 : tensor<8x6x6x18xbf16>

  // CHECK: "tf.BroadcastTo"
  // CHECK: "tf.Reshape"
}
