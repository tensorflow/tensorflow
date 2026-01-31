// RUN: litert-opt %s -tfl-identify-dilated-conv | FileCheck %s

func.func @testDilatedConv(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_0) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x120x120x8xf32>
  func.return %2 : tensor<1x120x120x8xf32>

  // CHECK-LABEL: testDilatedConv
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x120x120x8xf32>
}

func.func @testDilatedConvWithNonConstantPadAndCrops(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x64x64x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x64x64x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x60x60x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x60x60x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x120x120x8xf32>
  func.return %2 : tensor<1x120x120x8xf32>

  // CHECK-LABEL: testDilatedConvWithNonConstantPadAndCrops
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x120x120x8xf32>
}

func.func @testDilatedConvWithNonZeroBasePadding(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  func.return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithNonZeroBasePadding
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedConvWithFp16(%arg0 : tensor<1x20x30x40xf16>, %arg1: tensor<5x5x40x32xf16>) -> tensor<1x20x30x32xf16> {
  %block_shape = arith.constant dense<2> : tensor<2xi32>
  %paddings = arith.constant dense<4> : tensor<2x2xi32>
  %crops = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %block_shape, %paddings) : (tensor<1x20x30x40xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x14x19x40xf16>
  %1 = "tf.Conv2D"(%0, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x14x19x40xf16>, tensor<5x5x40x32xf16>) -> tensor<4x10x15x32xf16>
  %2 = "tf.BatchToSpaceND"(%1, %block_shape, %crops) : (tensor<4x10x15x32xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x20x30x32xf16>
  func.return %2 : tensor<1x20x30x32xf16>

  // CHECK-LABEL: testDilatedConvWithFp16
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x20x30x40xf16>, [[FILTER:%.*]]: tensor<5x5x40x32xf16>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{data_format = "NHWC", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x20x30x40xf16>, tensor<5x5x40x32xf16>) -> tensor<1x20x30x32xf16>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x20x30x32xf16>
}

func.func @testDilatedConvWithNonTrivialDilations(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", dilations = [1, 2, 2, 1], strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x60x60x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x60x60x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  func.return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithNonTrivialDilations
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func.func @testDilatedDepthWiseConv(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  func.return %2 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConv
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedConvWithPad(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<4x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.Pad"(%1, %cst_2) : (tensor<4x64x64x8xf32>, tensor<4x2xi32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.BatchToSpaceND"(%2, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %4 = "tf.BiasAdd"(%3, %arg2) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  func.return %4 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithPad
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedDepthWiseConvWithPad(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<4x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.Pad"(%1, %cst_2) : (tensor<4x64x64x8xf32>, tensor<4x2xi32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.BatchToSpaceND"(%2, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %4 = "tf.BiasAdd"(%3, %arg2) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  func.return %4 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithPad
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedConvWithBiasAdd(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %3 = "tf.BiasAdd"(%2, %arg2) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  func.return %3 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedConvWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedDepthWiseConvWithBiasAdd(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.DepthwiseConv2dNative"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x8xf32>
  %3 = "tf.BiasAdd"(%2, %arg2) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  func.return %3 : tensor<1x128x128x8xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[CONV]], [[BIAS]]) : (tensor<1x128x128x8xf32>, tensor<8xf32>) -> tensor<1x128x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128x8xf32>
}

func.func @testDilatedConvWithExpandSqueeze1(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %cst_2) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %5 = "tf.BiasAdd"(%4, %arg2) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  func.return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze1
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testDilatedDepthWiseConvWithExpandSqueeze1(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %cst_2) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %5 = "tf.BiasAdd"(%4, %arg2) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  func.return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze1
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testDilatedConvWithExpandSqueeze2(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<?xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x?x?xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x?x?xf32>, tensor<i32>) -> tensor<4x?x?x1xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x?x?x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x?x?x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x?x?x1xf32>) -> tensor<4x?x?xf32>
  %4 = "tf.BiasAdd"(%3, %arg2) : (tensor<4x?x?xf32>, tensor<?xf32>) -> tensor<4x?x?xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %cst_2) : (tensor<4x?x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  func.return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze2
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<?xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<?xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testDilatedDepthWiseConvWithExpandSqueeze2(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<?xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x?x?xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x?x?xf32>, tensor<i32>) -> tensor<4x?x?x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x?x?x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x?x?x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x?x?x1xf32>) -> tensor<4x?x?xf32>
  %4 = "tf.BiasAdd"(%3, %arg2) : (tensor<4x?x?xf32>, tensor<?xf32>) -> tensor<4x?x?xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %cst_2) : (tensor<4x?x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  func.return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze2
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<?xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<?xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testDilatedConvWithExpandSqueeze3(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %cst_3 = arith.constant dense<0> : tensor<3x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.Pad"(%3, %cst_3) : (tensor<4x64x64xf32>, tensor<3x2xi32>) -> tensor<4x64x64xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %cst_2) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %6 = "tf.BiasAdd"(%5, %arg2) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  func.return %6 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedConvWithExpandSqueeze3
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testDilatedDepthWiseConvWithExpandSqueeze3(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %cst_3 = arith.constant dense<0> : tensor<3x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.DepthwiseConv2dNative"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.Pad"(%3, %cst_3) : (tensor<4x64x64xf32>, tensor<3x2xi32>) -> tensor<4x64x64xf32>
  %5 = "tf.BatchToSpaceND"(%4, %cst, %cst_2) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %6 = "tf.BiasAdd"(%5, %arg2) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  func.return %6 : tensor<1x128x128xf32>

  // CHECK-LABEL: testDilatedDepthWiseConvWithExpandSqueeze3
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128xf32>, [[FILTER:%.*]]: tensor<5x5x1x1xf32>, [[BIAS:%.*]]: tensor<128xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x128xf32>, tensor<i32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.DepthwiseConv2dNative"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x1xf32>, tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [3]}> : (tensor<1x128x128x1xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x128xf32>
}

func.func @testAvoidDilatedConvWithExpand(%arg0: tensor<*xf32>, %arg1: tensor<5x5x1x1xf32>, %arg2: tensor<128xf32>) -> tensor<1x128x128xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<*xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [3]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %cst_2) : (tensor<4x64x64xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128xf32>
  %5 = "tf.BiasAdd"(%4, %arg2) : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  func.return %5 : tensor<1x128x128xf32>

  // CHECK-LABEL: testAvoidDilatedConvWithExpand
  // CHECK: "tf.SpaceToBatchND"
  // CHECK: "tf.ExpandDims"
  // CHECK: "tf.Conv2D"
  // CHECK: "tf.Squeeze"
  // CHECK: "tf.BatchToSpaceND"
  // CHECK: "tf.BiasAdd"
}

func.func @testDilatedConvWithDifferentExpandSqueezeAxis(%arg0: tensor<1x128x128xf32>, %arg1: tensor<5x5x1x1xf32>) -> tensor<1x128x128x1xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_1) : (tensor<1x128x128xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<4x68x68xf32>, tensor<i32>) -> tensor<4x68x68x1xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x1xf32>, tensor<5x5x1x1xf32>) -> tensor<4x64x64x1xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [2]} : (tensor<4x64x64x1xf32>) -> tensor<4x64x64x1xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst, %cst_2) : (tensor<4x64x64x1xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x1xf32>
  func.return %4 : tensor<1x128x128x1xf32>

  // CHECK-LABEL: testDilatedConvWithDifferentExpandSqueezeAxis
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func.func @testNoDilatedConvWhenFirstDimIsDynamic(%arg0: tensor<?x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<?x128x128x8xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<?x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x68x68x3xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<?x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<?x64x64x8xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<?x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x128x128x8xf32>
  func.return %2 : tensor<?x128x128x8xf32>

  // CHECK-LABEL: testNoDilatedConvWhenFirstDimIsDynamic
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func.func @testNoDilatedConvWhenLastDimIsDynamic(%arg0: tensor<1x128x128x?xf32>, %arg1: tensor<5x5x3x8xf32>) -> tensor<1x128x128x?xf32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %cst_1 = arith.constant dense<0> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x?xf32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x?xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x?xf32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_1) : (tensor<4x64x64x?xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x128x128x?xf32>
  func.return %2 : tensor<1x128x128x?xf32>

  // CHECK-LABEL: testNoDilatedConvWhenLastDimIsDynamic
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func.func @testNoDilatedConvWhenGivenInputIsNonFloatType(%arg0: tensor<1x128x128x3xi32>, %arg1: tensor<5x5x3x8xi32>) -> tensor<1x120x120x8xi32> {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xi32>
  %1 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xi32>, tensor<5x5x3x8xi32>) -> tensor<4x64x64x8xi32>
  %2 = "tf.BatchToSpaceND"(%1, %cst, %cst_0) : (tensor<4x64x64x8xi32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x120x120x8xi32>
  func.return %2 : tensor<1x120x120x8xi32>

  // CHECK-LABEL: testNoDilatedConvWhenGivenInputIsNonFloatType
  // CHECK: [[STB:%.*]] = "tf.SpaceToBatchND"
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BatchToSpaceND"
  // CHECK-NEXT: return [[RESULT]]
}

func.func @testDilatedConv1DExpandH(%arg0: tensor<1x128x3xf32>, %arg1: tensor<1x5x3x8xf32>) -> tensor<1x128x8xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<4> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst_1, %cst_2) : (tensor<1x128x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x68x3xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<2x68x3xf32>, tensor<i32>) -> tensor<2x1x68x3xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<2x1x68x3xf32>, tensor<1x5x3x8xf32>) -> tensor<2x1x64x8xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [-3]} : (tensor<2x1x64x8xf32>) -> tensor<2x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst_1, %cst) : (tensor<2x64x8xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x128x8xf32>
  func.return %4 : tensor<1x128x8xf32>

  // CHECK-LABEL: testDilatedConv1DExpandH
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x3xf32>, [[FILTER:%.*]]: tensor<1x5x3x8xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<-3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x3xf32>, tensor<i32>) -> tensor<1x1x128x3xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 1, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x1x128x3xf32>, tensor<1x5x3x8xf32>) -> tensor<1x1x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [-3]}> : (tensor<1x1x128x8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x8xf32>
}

func.func @testDilatedConv1DExpandHWithBiasAdd(%arg0: tensor<1x128x3xf32>, %arg1: tensor<1x5x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x8xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<4> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst_1, %cst_2) : (tensor<1x128x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x68x3xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<2x68x3xf32>, tensor<i32>) -> tensor<2x1x68x3xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<2x1x68x3xf32>, tensor<1x5x3x8xf32>) -> tensor<2x1x64x8xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [-3]} : (tensor<2x1x64x8xf32>) -> tensor<2x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst_1, %cst) : (tensor<2x64x8xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x128x8xf32>
  %5 = "tf.BiasAdd"(%4, %arg2) : (tensor<1x128x8xf32>, tensor<8xf32>) -> tensor<1x128x8xf32>
  func.return %5 : tensor<1x128x8xf32>

  // CHECK-LABEL: testDilatedConv1DExpandHWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x3xf32>, [[FILTER:%.*]]: tensor<1x5x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<-3> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x3xf32>, tensor<i32>) -> tensor<1x1x128x3xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 1, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x1x128x3xf32>, tensor<1x5x3x8xf32>) -> tensor<1x1x128x8xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [-3]}> : (tensor<1x1x128x8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x8xf32>, tensor<8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x8xf32>
}

func.func @testDilatedConv1DExpandW(%arg0: tensor<1x128x3xf32>, %arg1: tensor<5x1x3x8xf32>) -> tensor<1x128x8xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<-2> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<4> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst_1, %cst_2) : (tensor<1x128x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x68x3xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<2x68x3xf32>, tensor<i32>) -> tensor<2x68x1x3xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<2x68x1x3xf32>, tensor<5x1x3x8xf32>) -> tensor<2x64x1x8xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [-2]} : (tensor<2x64x1x8xf32>) -> tensor<2x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst_1, %cst) : (tensor<2x64x8xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x128x8xf32>
  func.return %4 : tensor<1x128x8xf32>

  // CHECK-LABEL: testDilatedConv1DExpandW
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x3xf32>, [[FILTER:%.*]]: tensor<5x1x3x8xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<-2> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x3xf32>, tensor<i32>) -> tensor<1x128x1x3xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x1x3xf32>, tensor<5x1x3x8xf32>) -> tensor<1x128x1x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [-2]}> : (tensor<1x128x1x8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x8xf32>
}

func.func @testDilatedConv1DExpandWWithBiasAdd(%arg0: tensor<1x128x3xf32>, %arg1: tensor<5x1x3x8xf32>, %arg2: tensor<8xf32>) -> tensor<1x128x8xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<-2> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<4> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst_1, %cst_2) : (tensor<1x128x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x68x3xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<2x68x3xf32>, tensor<i32>) -> tensor<2x68x1x3xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<2x68x1x3xf32>, tensor<5x1x3x8xf32>) -> tensor<2x64x1x8xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [-2]} : (tensor<2x64x1x8xf32>) -> tensor<2x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst_1, %cst) : (tensor<2x64x8xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x128x8xf32>
  %5 = "tf.BiasAdd"(%4, %arg2) : (tensor<1x128x8xf32>, tensor<8xf32>) -> tensor<1x128x8xf32>
  func.return %5 : tensor<1x128x8xf32>

  // CHECK-LABEL: testDilatedConv1DExpandWWithBiasAdd
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x3xf32>, [[FILTER:%.*]]: tensor<5x1x3x8xf32>, [[BIAS:%.*]]: tensor<8xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<-2> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x3xf32>, tensor<i32>) -> tensor<1x128x1x3xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 2, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x128x1x3xf32>, tensor<5x1x3x8xf32>) -> tensor<1x128x1x8xf32>
  // CHECK-NEXT: [[SQUEEZE:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [-2]}> : (tensor<1x128x1x8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.BiasAdd"([[SQUEEZE]], [[BIAS]]) : (tensor<1x128x8xf32>, tensor<8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x8xf32>
}

func.func @testDilatedConv1DWithMixedPostiveAndNegativeAxis(%arg0: tensor<1x128x3xf32>, %arg1: tensor<1x5x3x8xf32>) -> tensor<1x128x8xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<4> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst_1, %cst_2) : (tensor<1x128x3xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x68x3xf32>
  %1 = "tf.ExpandDims"(%0, %cst_0) : (tensor<2x68x3xf32>, tensor<i32>) -> tensor<2x1x68x3xf32>
  %2 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<2x1x68x3xf32>, tensor<1x5x3x8xf32>) -> tensor<2x1x64x8xf32>
  %3 = "tf.Squeeze"(%2) {squeeze_dims = [-3]} : (tensor<2x1x64x8xf32>) -> tensor<2x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%3, %cst_1, %cst) : (tensor<2x64x8xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x128x8xf32>
  func.return %4 : tensor<1x128x8xf32>

  // CHECK-LABEL: testDilatedConv1DWithMixedPostiveAndNegativeAxis
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x3xf32>, [[FILTER:%.*]]: tensor<1x5x3x8xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) : (tensor<1x128x3xf32>, tensor<i32>) -> tensor<1x1x128x3xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{dilations = [1, 1, 2, 1], padding = "SAME", strides = [1, 1, 1, 1]}> : (tensor<1x1x128x3xf32>, tensor<1x5x3x8xf32>) -> tensor<1x1x128x8xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [-3]}> : (tensor<1x1x128x8xf32>) -> tensor<1x128x8xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<1x128x8xf32>
}

func.func @testPaddedDilatedConv(%arg0 : tensor<2x1920x64xf32>) ->  tensor<2x1920x128xf32> {
  %0 = "tf.Const"() {value = dense<[[0, 0], [2, 0], [0, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %4 = "tf.Const"() {value = dense<0.0> : tensor<3x1x64x128xf32>} : () -> tensor<3x1x64x128xf32>
  %5 = "tf.SpaceToBatchND"(%arg0, %1, %3) {device = ""} : (tensor<2x1920x64xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<4x960x64xf32>
  %6 = "tf.ExpandDims"(%5, %2) {device = ""} : (tensor<4x960x64xf32>, tensor<i32>) -> tensor<4x960x1x64xf32>
  %7 = "tf.Conv2D"(%6, %4) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<4x960x1x64xf32>, tensor<3x1x64x128xf32>) -> tensor<4x958x1x128xf32>
  %8 = "tf.Squeeze"(%7) {device = "", squeeze_dims = [2]} : (tensor<4x958x1x128xf32>) -> tensor<4x958x128xf32>
  %9 = "tf.Pad"(%8, %0) {device = ""} : (tensor<4x958x128xf32>, tensor<3x2xi32>) -> tensor<4x960x128xf32>
  %10 = "tf.BatchToSpaceND"(%9, %1, %3) {device = ""} : (tensor<4x960x128xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<2x1920x128xf32>
  func.return %10 : tensor<2x1920x128xf32>

  // CHECK-LABEL: testPaddedDilatedConv
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<2x1920x64xf32>)
  // CHECK-NEXT: [[AXIS:%.*]] = "tf.Const"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  // CHECK-NEXT: [[FILTER:%.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<3x1x64x128xf32>}> : () -> tensor<3x1x64x128xf32>
  // CHECK-NEXT: [[EXPAND:%.*]] = "tf.ExpandDims"([[INPUT]], [[AXIS]]) {device = ""} : (tensor<2x1920x64xf32>, tensor<i32>) -> tensor<2x1920x1x64xf32>
  // CHECK-NEXT: [[CONV:%.*]] = "tf.Conv2D"([[EXPAND]], [[FILTER]]) <{data_format = "NHWC", dilations = [1, 2, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}> {device = ""} : (tensor<2x1920x1x64xf32>, tensor<3x1x64x128xf32>) -> tensor<2x1920x1x128xf32>
  // CHECK-NEXT: [[RESULT:%.*]] = "tf.Squeeze"([[CONV]]) <{squeeze_dims = [2]}> {device = ""} : (tensor<2x1920x1x128xf32>) -> tensor<2x1920x128xf32>
  // CHECK-NEXT: return [[RESULT]] : tensor<2x1920x128xf32>
}

func.func @testDilatedConvInterleaved(%arg0: tensor<1x128x128x3xf32>, %arg1: tensor<5x5x3x8xf32>) -> (tensor<1x120x120x8xf32>, tensor<1x120x120x8xf32>) {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %cst_0 = arith.constant dense<4> : tensor<2x2xi32>
  %0 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %1 = "tf.SpaceToBatchND"(%arg0, %cst, %cst_0) : (tensor<1x128x128x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x68x68x3xf32>
  %2 = "tf.Conv2D"(%0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %3 = "tf.Conv2D"(%1, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<4x68x68x3xf32>, tensor<5x5x3x8xf32>) -> tensor<4x64x64x8xf32>
  %4 = "tf.BatchToSpaceND"(%2, %cst, %cst_0) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x120x120x8xf32>
  %5 = "tf.BatchToSpaceND"(%3, %cst, %cst_0) : (tensor<4x64x64x8xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x120x120x8xf32>
  func.return %4, %5: tensor<1x120x120x8xf32>, tensor<1x120x120x8xf32>

  // CHECK-LABEL: testDilatedConvInterleaved
  // CHECK-SAME: ([[INPUT:%.*]]: tensor<1x128x128x3xf32>, [[FILTER:%.*]]: tensor<5x5x3x8xf32>)
  // CHECK-NEXT: [[RESULT0:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32>
  // CHECK-NEXT: [[RESULT1:%.*]] = "tf.Conv2D"([[INPUT]], [[FILTER]]) <{dilations = [1, 2, 2, 1], padding = "VALID", strides = [1, 1, 1, 1]}> : (tensor<1x128x128x3xf32>, tensor<5x5x3x8xf32>) -> tensor<1x120x120x8xf32>
  // CHECK-NEXT: return [[RESULT0]], [[RESULT1]] : tensor<1x120x120x8xf32>, tensor<1x120x120x8xf32>
}
