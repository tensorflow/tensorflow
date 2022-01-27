// RUN: tac-opt-all-backends -tfl-device-transform-gpu %s -split-input-file -verify-diagnostics | FileCheck %s

func @pack(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> {
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK:   func @pack(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<2x1xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant dense<1> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant dense<2> : tensor<1xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<[2, 1]> : tensor<2xi32>
// CHECK:           %[[VAL_5:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_2]]) : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tfl.reshape"(%[[VAL_1]], %[[VAL_2]]) : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.concatenation"(%[[VAL_5]], %[[VAL_6]]) {axis = 3 : i32, fused_activation_function = "NONE"} : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x2xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.reshape"(%[[VAL_7]], %[[VAL_3]]) : (tensor<1x1x1x2xf32>, tensor<1xi32>) -> tensor<2xf32>
// CHECK:           %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_8]], %[[VAL_4]]) : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_9]] : tensor<2x1xf32>
// CHECK:         }

// -----

// CHECK-LABEL: @avoidPackToConcatenationOnUnknownShapes
func @avoidPackToConcatenationOnUnknownShapes(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x1xf32> {
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x1xf32>
  return %0 : tensor<?x?x1xf32>
}

// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.pack"

// -----

// CHECK-LABEL: @avoidPackToConcatenationOnUnknownRank
func @avoidPackToConcatenationOnUnknownRank(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.pack"

// -----

func @squaredDifference(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK:       func @squaredDifference(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:         %0 = "tf.Sub"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         %1 = "tf.Mul"(%0, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         return %1 : tensor<4xf32>
// CHECK:       }

// -----

func @unrollSplit(%arg0: tensor<i32>, %arg1: tensor<1x8x8x1024xf32>) -> (tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>) {
  %0:4 = "tfl.split"(%arg0, %arg1) {num_splits = 4 : i32, tac.device = "CPU"} : (tensor<i32>, tensor<1x8x8x1024xf32>) -> (tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>)
  return %0#0, %0#1, %0#3 : tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>
}

// CHECK:        func @unrollSplit([[VAL_0:%.*]]: tensor<i32>, [[VAL_1:%.*]]: tensor<1x8x8x1024xf32>) -> (tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>) {
// CHECK-DAG:       [[VAL_2:%.*]] = arith.constant dense<0> : tensor<4xi32>
// CHECK-DAG:       [[VAL_3:%.*]] = arith.constant dense<[0, 0, 0, 256]> : tensor<4xi32>
// CHECK-DAG:       [[VAL_4:%.*]] = arith.constant dense<[0, 0, 0, 768]> : tensor<4xi32>
// CHECK-DAG:       [[VAL_5:%.*]] = arith.constant dense<[-1, -1, -1, 256]> : tensor<4xi32>
// CHECK:           [[VAL_6:%.*]] = "tfl.slice"([[VAL_1]], [[VAL_2]], [[VAL_5]]) : (tensor<1x8x8x1024xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x8x8x256xf32>
// CHECK:           [[VAL_7:%.*]] = "tfl.slice"([[VAL_1]], [[VAL_3]], [[VAL_5]]) : (tensor<1x8x8x1024xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x8x8x256xf32>
// CHECK:           [[VAL_8:%.*]] = "tfl.slice"([[VAL_1]], [[VAL_4]], [[VAL_5]]) : (tensor<1x8x8x1024xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x8x8x256xf32>
// CHECK:           return [[VAL_6]], [[VAL_7]], [[VAL_8]] : tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>, tensor<1x8x8x256xf32>
// CHECK:         }

// -----

func @unrollSplitUnknownRankResults(%arg0: tensor<i32>, %arg1: tensor<1x8x8x1024xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %0:4 = "tfl.split"(%arg0, %arg1) {num_splits = 4 : i32, tac.device = "CPU"} : (tensor<i32>, tensor<1x8x8x1024xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1, %0#3 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @unrollSplitUnknownRankResults
// CHECK-NOT: "tfl.slice"
// CHECK: "tfl.split"

// -----

func @unrollSplitV(%arg0: tensor<?x13x13x85xf32>) -> (tensor<?x13x13x2xf32>, tensor<?x13x13x2xf32>, tensor<?x13x13x81xf32>) {
  %0 = "tfl.pseudo_const"() {value = dense<[2, 2, 81]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %2:3 = "tfl.split_v"(%arg0, %0, %1) {num_splits = 3 : i32} : (tensor<?x13x13x85xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<?x13x13x2xf32>, tensor<?x13x13x2xf32>, tensor<?x13x13x81xf32>)
  return %2#0, %2#1, %2#2 : tensor<?x13x13x2xf32>, tensor<?x13x13x2xf32>, tensor<?x13x13x81xf32>
}

// CHECK:   func @unrollSplitV(%[[VAL_0:.*]]: tensor<?x13x13x85xf32>) -> (tensor<?x13x13x2xf32>, tensor<?x13x13x2xf32>, tensor<?x13x13x81xf32>) {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant dense<0> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant dense<[0, 0, 0, 2]> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant dense<[-1, -1, -1, 2]> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant dense<[0, 0, 0, 4]> : tensor<4xi32>
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant dense<[-1, -1, -1, 81]> : tensor<4xi32>
// CHECK:           %[[VAL_6:.*]] = "tfl.slice"(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]) : (tensor<?x13x13x85xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x13x13x2xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.slice"(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]]) : (tensor<?x13x13x85xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x13x13x2xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.slice"(%[[VAL_0]], %[[VAL_4]], %[[VAL_5]]) : (tensor<?x13x13x85xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x13x13x81xf32>
// CHECK:           return %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : tensor<?x13x13x2xf32>, tensor<?x13x13x2xf32>, tensor<?x13x13x81xf32>
// CHECK:         }

// -----

func @unrollSplitVUnknownRankResults(%arg0: tensor<?x13x13x85xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
  %0 = "tfl.pseudo_const"() {value = dense<[2, 2, 81]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %2:3 = "tfl.split_v"(%arg0, %0, %1) {num_splits = 3 : i32} : (tensor<?x13x13x85xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %2#0, %2#1, %2#2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @unrollSplitVUnknownRankResults
// CHECK-NOT: "tfl.slice"
// CHECK: "tfl.split_v"

// -----

func @sub(%arg0: tensor<1x384x384x3xf32>, %arg1: tensor<3xf32>) -> tensor<1x384x384x3xf32> {
  %0 = "tfl.sub"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x384x384x3xf32>, tensor<3xf32>) -> tensor<1x384x384x3xf32>
  return %0 : tensor<1x384x384x3xf32>
}

// CHECK:       func @sub(%[[VAL_0:.*]]: tensor<1x384x384x3xf32>, %[[VAL_1:.*]]: tensor<3xf32>) -> tensor<1x384x384x3xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<-1.000000e+00> : tensor<f32>
// CHECK:           %[[VAL_3:.*]] = tfl.mul(%[[VAL_1]], %[[VAL_2]]) {fused_activation_function = "NONE"} : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK:           %[[VAL_4:.*]] = tfl.add(%[[VAL_0]], %[[VAL_3]]) {fused_activation_function = "NONE"} : (tensor<1x384x384x3xf32>, tensor<3xf32>) -> tensor<1x384x384x3xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x384x384x3xf32>
// CHECK:         }

// -----

func @ensureBiasForConv2d(%arg0: tensor<128x32x32x3xf32>, %arg1: tensor<32x1x1x3xf32>) -> tensor<128x32x32x32xf32> {
  %cst = constant unit
  %0 = "tfl.conv_2d"(%arg0, %arg1, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<128x32x32x3xf32>, tensor<32x1x1x3xf32>, none) -> tensor<128x32x32x32xf32>
  return %0 : tensor<128x32x32x32xf32>
}

// CHECK:       func @ensureBiasForConv2d(%[[VAL_0:.*]]: tensor<128x32x32x3xf32>, %[[VAL_1:.*]]: tensor<32x1x1x3xf32>) -> tensor<128x32x32x32xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : tensor<32xf32>
// CHECK:           %[[VAL_3:.*]] = "tfl.conv_2d"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<128x32x32x3xf32>, tensor<32x1x1x3xf32>, tensor<32xf32>) -> tensor<128x32x32x32xf32>
// CHECK:           return %[[VAL_3]] : tensor<128x32x32x32xf32>
// CHECK:         }

// -----

func @padSliceTo4D(%arg0: tensor<4x384x32xf32>) -> tensor<1x384x32xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %2 : tensor<1x384x32xf32>
}

// CHECK:       func @padSliceTo4D(%[[VAL_0:.*]]: tensor<4x384x32xf32>) -> tensor<1x384x32xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:       %[[VAL_2:.*]] = "tf.Const"() {value = dense<[1, 1, 384, 32]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant dense<[1, 4, 384, 32]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<[1, 384, 32]> : tensor<3xi32>
// CHECK:           %[[VAL_5:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_3]]) : (tensor<4x384x32xf32>, tensor<4xi32>) -> tensor<1x4x384x32xf32>
// CHECK:           %[[VAL_6:.*]] = "tfl.slice"(%[[VAL_5]], %[[VAL_1]], %[[VAL_2]]) : (tensor<1x4x384x32xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x384x32xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_6]], %[[VAL_4]]) : (tensor<1x1x384x32xf32>, tensor<3xi32>) -> tensor<1x384x32xf32>
// CHECK:           return %[[VAL_7]] : tensor<1x384x32xf32>
// CHECK:         }

// -----

// CHECK-LABEL: @avoidPadSliceTo4DOnUnknownOutputShape
func @avoidPadSliceTo4DOnUnknownOutputShape(%arg0: tensor<4x384x32xf32>) -> tensor<1x?x?xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x?x?xf32>
  return %2 : tensor<1x?x?xf32>
}

// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.slice"

// -----

func @fullyConnectedToConv(%arg0: tensor<384x384xf32>, %arg1: tensor<512x384xf32>, %arg2: tensor<512xf32>) -> tensor<384x512xf32> {
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x384xf32>, tensor<512x384xf32>, tensor<512xf32>) -> tensor<384x512xf32>
  return %0: tensor<384x512xf32>
}

// CHECK:       func @fullyConnectedToConv(%[[VAL_0:.*]]: tensor<384x384xf32>, %[[VAL_1:.*]]: tensor<512x384xf32>, %[[VAL_2:.*]]: tensor<512xf32>) -> tensor<384x512xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant dense<[1, 1, 384, 384]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<[512, 1, 1, 384]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant dense<[384, 512]> : tensor<2xi32>
// CHECK:           %[[VAL_6:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_3]]) : (tensor<384x384xf32>, tensor<4xi32>) -> tensor<1x1x384x384xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_1]], %[[VAL_4]]) : (tensor<512x384xf32>, tensor<4xi32>) -> tensor<512x1x1x384xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.conv_2d"(%[[VAL_6]], %[[VAL_7]], %[[VAL_2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1x384x384xf32>, tensor<512x1x1x384xf32>, tensor<512xf32>) -> tensor<1x1x384x512xf32>
// CHECK:           %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_8]], %[[VAL_5]]) : (tensor<1x1x384x512xf32>, tensor<2xi32>) -> tensor<384x512xf32>
// CHECK:           return %[[VAL_9]] : tensor<384x512xf32>
// CHECK:         }

// -----

func @padConcatTo4D(%arg0: tensor<384x384xf32>, %arg1: tensor<384x384xf32>, %arg2: tensor<384x384xf32>, %arg3: tensor<384x384xf32>) -> tensor<1536x384xf32> {
 %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<384x384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384x384xf32>) -> tensor<1536x384xf32>
  return %0: tensor<1536x384xf32>
}

// CHECK:   func @padConcatTo4D(%[[VAL_0:.*]]: tensor<384x384xf32>, %[[VAL_1:.*]]: tensor<384x384xf32>, %[[VAL_2:.*]]: tensor<384x384xf32>, %[[VAL_3:.*]]: tensor<384x384xf32>) -> tensor<1536x384xf32> {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<[1, 1, 384, 384]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant dense<[1536, 384]> : tensor<2xi32>
// CHECK:           %[[VAL_6:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_4]]) : (tensor<384x384xf32>, tensor<4xi32>) -> tensor<1x1x384x384xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_1]], %[[VAL_4]]) : (tensor<384x384xf32>, tensor<4xi32>) -> tensor<1x1x384x384xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.reshape"(%[[VAL_2]], %[[VAL_4]]) : (tensor<384x384xf32>, tensor<4xi32>) -> tensor<1x1x384x384xf32>
// CHECK:           %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_3]], %[[VAL_4]]) : (tensor<384x384xf32>, tensor<4xi32>) -> tensor<1x1x384x384xf32>
// CHECK:           %[[VAL_10:.*]] = "tfl.concatenation"(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<1x1x384x384xf32>, tensor<1x1x384x384xf32>, tensor<1x1x384x384xf32>, tensor<1x1x384x384xf32>) -> tensor<1x1x1536x384xf32>
// CHECK:           %[[VAL_11:.*]] = "tfl.reshape"(%[[VAL_10]], %[[VAL_5]]) : (tensor<1x1x1536x384xf32>, tensor<2xi32>) -> tensor<1536x384xf32>
// CHECK:           return %[[VAL_11]] : tensor<1536x384xf32>
// CHECK:         }

// -----

// CHECK-LABEL: @avoidPadConcatTo4DOnUnknownOutputShape
func @avoidPadConcatTo4DOnUnknownOutputShape(%arg0: tensor<384x384xf32>, %arg1: tensor<384x384xf32>, %arg2: tensor<384x384xf32>, %arg3: tensor<384x384xf32>) -> tensor<?x?xf32> {
 %0 = "tfl.concatenation"(%arg0, %arg1, %arg2, %arg3) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<384x384xf32>, tensor<384x384xf32>, tensor<384x384xf32>, tensor<384x384xf32>) -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}

// CHECK-NOT: "tfl.reshape"
// CHECK: "tfl.concatenation"
