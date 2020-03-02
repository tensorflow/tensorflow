// RUN: tf-opt %s -tfl-identify-dilated-conv | FileCheck %s --dump-input-on-failure

func @testDilatedConv(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConv
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedConvWithNonZeroSTBPadding(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = constant dense<2> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithNonZeroSTBPadding
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedConvWithNonTrivialDilations(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg2) {padding = "VALID", dilations = [1, 2, 2, 1], strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithNonTrivialDilations
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func @testDilatedDepthWiseConv(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConv
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedConvWithPad(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>, %arg3: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.Pad"(%1, %arg1) : (tensor<4x64x64x8xf32>, tensor<2x2xi32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.BatchToSpaceND"(%2, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %4 = "tf.BiasAdd"(%3, %arg3) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  return %4 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithPad
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedDepthWiseConvWithPad(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>, %arg3: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.Pad"(%1, %arg1) : (tensor<4x64x64x8xf32>, tensor<2x2xi32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.BatchToSpaceND"(%2, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %4 = "tf.BiasAdd"(%3, %arg3) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  return %4 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithPad
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedConvWithBiasAdd(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>, %arg3: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %3 = "tf.BiasAdd"(%2, %arg3) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  return %3 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedDepthWiseConvWithBiasAdd(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x3x8xf32>, %arg3: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %arg1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %3 = "tf.BiasAdd"(%2, %arg3) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  return %3 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func @testDilatedConvWithExpandSqueeze1(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %arg1) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %5 = "tf.BiasAdd"(%4, %arg3) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze1
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedDepthWiseConvWithExpandSqueeze1(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %arg1) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %5 = "tf.BiasAdd"(%4, %arg3) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze1
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedConvWithExpandSqueeze2(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<?xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x?x?xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x?x?xf32>, tensor<i32>) -> tensor<4x?x?x1xf32>
  %2 = "tf.Conv2D"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x?x?x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x?x?x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x?x?x1xf32>) -> tensor<4x?x?xf32>
  %4 = "tf.BiasAdd"(%3, %arg3) : (tensor<4x?x?xf32>, tensor<?xf32>) -> tensor<4x?x?xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %arg1) : (tensor<4x?x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze2
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<?xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<?xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedDepthWiseConvWithExpandSqueeze2(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<?xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x?x?xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x?x?xf32>, tensor<i32>) -> tensor<4x?x?x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x?x?x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x?x?x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x?x?x1xf32>) -> tensor<4x?x?xf32>
  %4 = "tf.BiasAdd"(%3, %arg3) : (tensor<4x?x?xf32>, tensor<?xf32>) -> tensor<4x?x?xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %arg1) : (tensor<4x?x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze2
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<?xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<?xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedConvWithExpandSqueeze3(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.Pad"(%3, %arg1) : (tensor<4x64x64xf32>, tensor<2x2xi32>) -> tensor<4x64x64xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %arg1) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %6 = "tf.BiasAdd"(%5, %arg3) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  return %6 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze3
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedDepthWiseConvWithExpandSqueeze3(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.Pad"(%3, %arg1) : (tensor<4x64x64xf32>, tensor<2x2xi32>) -> tensor<4x64x64xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %arg1) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %6 = "tf.BiasAdd"(%5, %arg3) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  return %6 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze3
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[PADDING:%.*]]: tensor<2x2xi32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) {dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) {squeeze_dims = [3]} : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func @testDilatedConvWithDifferentExpandSqueezeAxis(%arg0: tensor<1x128x128xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<5x5x1x1xf32>, %arg3: tensor<128xf32>) -> tensor<1x128x128x1xf32> {
  %cst = constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %arg1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg2) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [2]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64x1xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %arg1) : (tensor<4x64x64x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x1xf32>
  return %4 : tensor<1x128x128x1xf32>

  // CHECK-LABEL: testDilatedConvWithDifferentExpandSqueezeAxis
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}
