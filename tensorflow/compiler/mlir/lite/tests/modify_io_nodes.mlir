// RUN: tf-opt %s -tfl-modify-io-nodes="test-io-types=float32,float32" | FileCheck %s
// RUN: tf-opt %s -tfl-modify-io-nodes="test-io-types=int8,int8" | FileCheck --check-prefix=INT8 %s
// RUN: tf-opt %s -tfl-modify-io-nodes="test-io-types=uint8,uint8" | FileCheck --check-prefix=UINT8 %s

func.func @modified(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input", outputs = "output"}} {
  %cst = arith.constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  func.return %6 : tensor<1x401408xf32>

// CHECK-LABEL: func @modified(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32>
// CHECK-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
// CHECK-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// CHECK-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[q]], %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// CHECK-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// CHECK-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[softmax]]) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
// CHECK-NEXT: return %[[dq]] : tensor<1x401408xf32>

// INT8-LABEL: @modified(%arg0: tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// INT8-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// INT8-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// INT8-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// INT8-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// INT8-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// INT8-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// INT8-NEXT: return %[[softmax]] : tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>

// UINT8-LABEL: func @modified(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>
// UINT8-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// UINT8-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
// UINT8-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// UINT8-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// UINT8-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[q]], %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// UINT8-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// UINT8-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// UINT8-NEXT: %[[dq:.*]] = "tfl.quantize"(%[[softmax]]) {qtype = tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>} : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>
// UINT8-NEXT: return %[[dq]] : tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>
}

func.func @not_modified(%arg0: tensor<f32>, %arg1: tensor<1x224x224x3xf32>) -> (tensor<1x401408xf32>, tensor<1x224x224x3xf32>) attributes {tf.entry_function = {control_outputs = "", inputs = "input0,input1", outputs = "output0,output1"}} {
  %cst = arith.constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg1) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  func.return %6, %arg1 : tensor<1x401408xf32>, tensor<1x224x224x3xf32>

// CHECK-LABEL: func @not_modified(%arg0: tensor<f32>, %arg1: tensor<1x224x224x3xf32>) -> (tensor<1x401408xf32>, tensor<1x224x224x3xf32>)
// CHECK-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
// CHECK-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// CHECK-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// CHECK-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[q]], %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// CHECK-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// CHECK-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[softmax]]) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
// CHECK-NEXT: return %[[dq]], %arg1 : tensor<1x401408xf32>, tensor<1x224x224x3xf32>

// INT8-LABEL: @not_modified(%arg0: tensor<f32>, %arg1: tensor<1x224x224x3xf32>) -> (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>, tensor<1x224x224x3xf32>)
// INT8-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// INT8-NEXT: %[[q:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
// INT8-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// INT8-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// INT8-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[q]], %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// INT8-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// INT8-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// INT8-NEXT: return %[[softmax]], %arg1 : tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>, tensor<1x224x224x3xf32>

// UINT8-LABEL: func @not_modified(%arg0: tensor<f32>, %arg1: tensor<1x224x224x3xf32>) -> (tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>, tensor<1x224x224x3xf32>)
// UINT8-NEXT: %[[shape:.*]] = arith.constant dense<[1, 401408]> : tensor<2xi32>
// UINT8-NEXT: %[[q:.*]] = "tfl.quantize"(%arg1) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
// UINT8-NEXT: %[[cst1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
// UINT8-NEXT: %[[cst2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
// UINT8-NEXT: %[[conv:.*]] = "tfl.conv_2d"(%[[q]], %[[cst1]], %[[cst2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
// UINT8-NEXT: %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[shape]]) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
// UINT8-NEXT: %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// UINT8-NEXT: %[[dq:.*]] = "tfl.quantize"(%[[softmax]]) {qtype = tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>} : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>
// UINT8-NEXT: return %[[dq]], %arg1 : tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>, tensor<1x224x224x3xf32>
}

func.func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32> {
  %cst = arith.constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  func.return %6 : tensor<1x401408xf32>

// CHECK-LABEL: func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32>
// INT8-LABEL: @main(%arg0: tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
// UINT8-LABEL: func @main(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03:128>>
}

func.func @non_entry_funciton(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32> {
  %cst = arith.constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<i8:f32, 7.812500e-03>>, tensor<32x3x3x3x!quant.uniform<i8<-127:127>:f32, 0.021826678373682216>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<i8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<i8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<i8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  func.return %6 : tensor<1x401408xf32>

// CHECK-LABEL: func @non_entry_funciton(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32>
// INT8-LABEL: func @non_entry_funciton(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32>
// UINT8-LABEL: func @non_entry_funciton(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32>
}
