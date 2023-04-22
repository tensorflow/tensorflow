// RUN: tf-opt %s -tfl-post-quantize | FileCheck %s
// RUN: tf-opt %s -tfl-post-quantize-remove-qdq | FileCheck --check-prefix=QDQ %s

// CHECK-LABEL: RemoveUnused
// QDQ-LABEL: RemoveUnused
func @RemoveUnused(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>
  %1:4 = "tfl.split"(%arg1, %0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.0>>)
  -> (tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>,tensor<2x!quant.uniform<u8:f32, 1.0>>, tensor<2x!quant.uniform<u8:f32, 1.0>>)
  %2 = "tfl.dequantize"(%1#0) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  %3 = "tfl.dequantize"(%1#1) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>

  // unused quantization ops should be removed as well.
  %4 = "tfl.dequantize"(%1#2) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>} : (tensor<2xf32>) -> (tensor<2x!quant.uniform<u8:f32, 1.0>>)
  %6 = tfl.add %5, %5 {fused_activation_function = "NONE"} : tensor<2x!quant.uniform<u8:f32, 1.0>>

  return %2, %3 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:4 = "tfl.split"(%arg1, %arg0)
// CHECK-NEXT: return %[[split]]#0, %[[split]]#1

// QDQ-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) {qtype = tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>
// QDQ-NEXT: %[[split:.*]]:4 = "tfl.split"(%arg1, %[[q]]) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>)
// QDQ-NEXT: %[[out1:.*]] = "tfl.dequantize"(%[[split]]#0) : (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<2xf32>
// QDQ-NEXT: %[[out2:.*]] = "tfl.dequantize"(%[[split]]#1) : (tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<2xf32>
// QDQ-NEXT: return %[[out1]], %[[out2]] : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: RemoveTrival
// QDQ-LABEL: RemoveTrival
func @RemoveTrival(%arg0: tensor<384x512x!quant.uniform<i8:f32, 1.0:-128>>, %arg1: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.0>>, %arg2: none) -> tensor<384x128x!quant.uniform<i8:f32, 2.0>> {
  %1 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.0:-128>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.0>>, none) -> tensor<384x128x!quant.uniform<i8:f32, 1.0>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<384x128x!quant.uniform<i8:f32, 2.0>>} : (tensor<384x128x!quant.uniform<i8:f32, 1.0>>) -> tensor<384x128x!quant.uniform<i8:f32, 2.0>>
  return %2 : tensor<384x128x!quant.uniform<i8:f32, 2.0>>

// CHECK-NEXT: %[[fc:.*]] = "tfl.fully_connected"{{.*}} -> tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
// CHECK-NEXT: return %[[fc]]

// QDQ-NEXT: %[[fc:.*]] = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, none) -> tensor<384x128x!quant.uniform<i8:f32, 1.000000e+00>>
// QDQ-NEXT: %[[q:.*]] = "tfl.quantize"(%[[fc]]) {qtype = tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>} : (tensor<384x128x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
// QDQ-NEXT: return %[[q]] : tensor<384x128x!quant.uniform<i8:f32, 2.000000e+00>>
}

func @main(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x401408xf32> {
  %cst = constant dense<[1, 401408]> : tensor<2xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %2 = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>} : () -> tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>
  %3 = "tfl.conv_2d"(%0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %4 = "tfl.reshape"(%3, %cst) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>) -> tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>
  %5 = "tfl.softmax"(%4) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>
  %6 = "tfl.dequantize"(%5) : (tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<1x401408xf32>
  return %6 : tensor<1x401408xf32>
}

func @main2(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %1 = "tfl.quantize"(%arg1) {qtype = tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>} : (tensor<2x4xf32>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %2 = tfl.add %0, %1 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4xf32>
  return %3 : tensor<2x4xf32>
}

// CHECK: func @main(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK-NEXT:  %[[cst:.*]] = constant dense<[1, 401408]> : tensor<2xi32>
// CHECK-NEXT:  %[[q_cst_0:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>}
// CHECK-NEXT:  %[[q_cst_1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<i32:f32, 1.7052092479439231E-4>>, value = dense<0> : tensor<32xi32>}
// CHECK-NEXT:  %[[conv:.*]] = "tfl.conv_2d"(%arg0, %[[q_cst_0]], %[[q_cst_1]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32}
// CHECK-NEXT:  %[[reshape:.*]] = "tfl.reshape"(%[[conv]], %[[cst]]) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>, tensor<2xi32>)
// CHECK-NEXT:  %[[softmax:.*]] = "tfl.softmax"(%[[reshape]]) {beta = 1.000000e+00 : f32} : (tensor<1x401408x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK-NEXT:  return %[[softmax]] : tensor<1x401408x!quant.uniform<u8:f32, 3.906250e-03>>
// CHECK-NEXT:}

// CHECK: func @main2(%arg0: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>, %arg1: tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>) -> tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>> {
// CHECK-NEXT:  %[[add:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:  return %[[add]] : tensor<2x4x!quant.uniform<u8:f32, 0.49803921568627452>>
// CHECK-NEXT:}

// CHECK-LABEL: HandleReturnedDequantizeWithAnotherUse
func @HandleReturnedDequantizeWithAnotherUse(%arg0: tensor<128x16xf32>) -> (tensor<128x16xf32>, tensor<128xi32>) {
// CHECK-NEXT:  %[[cst:.*]] = constant dense<1> : tensor<i32>
  %cst = constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[softmax:.*]] = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<128x16xf32>) -> tensor<128x16xf32>
  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<128x16xf32>) -> tensor<128x16xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>, volatile} : (tensor<128x16xf32>) -> tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>
  %2 = "tfl.dequantize"(%1) : (tensor<128x16x!quant.uniform<u8:f32, 3.906250e-03>>) -> tensor<128x16xf32>
// CHECK-NEXT:  %[[argmax:.*]] = "tfl.arg_max"(%[[softmax]], %[[cst]]) : (tensor<128x16xf32>, tensor<i32>) -> tensor<128xi32>
  %3 = "tfl.arg_max"(%2, %cst) : (tensor<128x16xf32>, tensor<i32>) -> tensor<128xi32>
// CHECK-NEXT:  return %[[softmax]], %[[argmax]] : tensor<128x16xf32>, tensor<128xi32>
  return %2, %3 : tensor<128x16xf32>, tensor<128xi32>
}

// CHECK-LABEL: PruneUnusedLstm
func @PruneUnusedLstm(%arg0: tensor<1x28x28xf32>) -> (tensor<1x28x28xf32>) {
    %input = "tfl.quantize"(%arg0) {qtype = tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x28x28xf32>) -> tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>
    %cst_1 = "tfl.pseudo_qconst"() {qtype = tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<1x20xi8>} : () -> tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_2 = constant unit
    %cst_3 = "tfl.pseudo_qconst"() {qtype = tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20x20xi8>} : () -> tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_7 = "tfl.pseudo_qconst"() {qtype = tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20xi8>} : () -> tensor<20x!quant.uniform<i8:f32, 0.006:-34>>
    %cst_11 = "tfl.pseudo_qconst"() {qtype = tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, value = dense<1> : tensor<20x28xi8>} : () -> tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>
    %cell_input = "tfl.pseudo_qconst"() {qtype = tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>, value = dense<1> : tensor<1x20xi6>} : () -> tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>
    %0 = "tfl.unidirectional_sequence_lstm"(%input,
      %cst_11, %cst_11, %cst_11, %cst_11,
      %cst_3, %cst_3, %cst_3, %cst_3,
      %cst_2, %cst_2, %cst_2,
      %cst_7, %cst_7, %cst_7, %cst_7,
      %cst_2, %cst_2,
      %cst_1, %cell_input,
      %cst_2, %cst_2, %cst_2, %cst_2) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", proj_clip = 0.000000e+00 : f32, time_major = false}
    :  ( tensor<1x28x28x!quant.uniform<i8:f32, 0.003:-128>>,
         tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x28x!quant.uniform<i8:f32, 0.006:-34>>,
         tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x20x!quant.uniform<i8:f32, 0.006:-34>>,
         none, none, none,
         tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<20x!quant.uniform<i8:f32, 0.006:-34>>,
         none, none,
         tensor<1x20x!quant.uniform<i8:f32, 0.006:-34>>, tensor<1x20x!quant.uniform<i16:f32, 0.006:-34>>,
         none, none, none, none) -> tensor<1x28x20x!quant.uniform<i8:f32, 0.006:-34>>
    return %arg0 : tensor<1x28x28xf32>
// CHECK-NEXT: return %arg0
}

// CHECK-LABEL: HandleVolatileRequantizeOp
func @HandleVolatileRequantizeOp(%arg0: tensor<1x3x3xf32>) -> (tensor<1x3x3xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x3x3xf32>) -> tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.logistic"(%0) : (tensor<1x3x3x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x3x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>
  %2 = "tfl.quantize"(%1) {qtype = tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>, volatile} : (tensor<1x3x3x!quant.uniform<i8:f32, 3.906250e-03:-128>>) -> tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>
  %3 = "tfl.dequantize"(%2) : (tensor<1x3x3x!quant.uniform<i8:f32, 0.004:-128>>) -> tensor<1x3x3xf32>
  %4 = "tfl.div"(%arg0, %3) {fused_activation_function = "NONE"} : (tensor<1x3x3xf32>, tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  return %4 : tensor<1x3x3xf32>
//  CHECK: %[[logistic:.*]] = "tfl.logistic"
//  CHECK: %[[dq:.*]] = "tfl.dequantize"(%[[logistic]])
//  CHECK: %[[div:.*]] = tfl.div %arg0, %[[dq]]
}

// CHECK-LABEL: RemoveLeadingQdq
// QDQ-LABEL: RemoveLeadingQdq
func @RemoveLeadingQdq(%arg0: tensor<4xf32>, %arg1: tensor<i32>) -> (tensor<2xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<4x!quant.uniform<u8:f32, 1.0>>, volatile} : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<4x!quant.uniform<u8:f32, 1.0>>) -> tensor<4xf32>
  %2:4 = "tfl.split"(%arg1, %1) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>)
  -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %3 = "tfl.quantize"(%2#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.0>>, volatile} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.0>>
  %4 = "tfl.dequantize"(%3) : (tensor<2x!quant.uniform<u8:f32, 1.0>>) -> tensor<2xf32>
  return %4 : tensor<2xf32>

// CHECK-NEXT:  %[[dequant:.*]] = "tfl.dequantize"(%arg0) : (tensor<4x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<4xf32>
// CHECK-NEXT:  %[[split:.*]]:4 = "tfl.split"(%arg1, %[[dequant]]) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
// CHECK-NEXT:  %[[quant:.*]] = "tfl.quantize"(%[[split]]#0) {qtype = tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>, volatile} : (tensor<2xf32>) -> tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK-NEXT:  return %[[quant]] : tensor<2x!quant.uniform<u8:f32, 1.000000e+00>>

// QDQ-NEXT:  %[[split:.*]]:4 = "tfl.split"(%arg1, %arg0) {num_splits = 4 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
// QDQ-NEXT:  return %[[split]]#0 : tensor<2xf32>
}
