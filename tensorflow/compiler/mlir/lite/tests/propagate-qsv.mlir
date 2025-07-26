// RUN: litert-opt %s -tfl-propagate-qsv | FileCheck %s


// CHECK-LABEL: concat
// Checks that the QDQ is propagated from the operands to the output of concat.
func.func @concat(%arg0: tensor<2x1xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2x4xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x1x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<i16:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x1xf32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<2x3x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i16:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x3x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x3xf32>)
  %4 = "tfl.concatenation"(%1, %3) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  func.return %4: tensor<2x4xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[q_0:.*]] = "tfl.quantize"(%arg1)
// CHECK-NEXT: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK-NEXT: %[[c:.*]] = "tfl.concatenation"(%[[dq]], %[[dq_0]])
// CHECK-NEXT: %[[q_1:.*]] = "tfl.quantize"(%[[c]])
// CHECK-NEXT: %[[dq_1:.*]] = "tfl.dequantize"(%[[q_1]])
// CHECK-NEXT: return %[[dq_1:.*]]
}

// -----

// CHECK-LABEL: partial_quantized
func.func @partial_quantized(%arg0: tensor<2x1xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x4xf32>) -> (tensor<2x4xf32>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<2x1x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x1xf32>) -> tensor<2x1x!quant.uniform<i16:f32, 1.0>>
  %1 = "tfl.dequantize"(%0) : (tensor<2x1x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x1xf32>)
  %2 = "tfl.quantize"(%arg1) {qtype = tensor<2x3x!quant.uniform<i16:f32, 1.0>>} : (tensor<2x3xf32>) -> tensor<2x3x!quant.uniform<i16:f32, 1.0>>
  %3 = "tfl.dequantize"(%2) : (tensor<2x3x!quant.uniform<i16:f32, 1.0>>) -> (tensor<2x3xf32>)
  %4 = "tfl.concatenation"(%1, %3) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2x1xf32>, tensor<2x3xf32>) -> tensor<2x4xf32>
  %5 = "tfl.add"(%4, %arg2) {fused_activation_function = "NONE"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %5: tensor<2x4xf32>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0)
// CHECK-NEXT: %[[dq:.*]] = "tfl.dequantize"(%[[q]])
// CHECK-NEXT: %[[q_0:.*]] = "tfl.quantize"(%arg1)
// CHECK-NEXT: %[[dq_0:.*]] = "tfl.dequantize"(%[[q_0]])
// CHECK-NEXT: %[[c:.*]] = "tfl.concatenation"(%[[dq]], %[[dq_0]])
// CHECK-NEXT: %[[q_1:.*]] = "tfl.quantize"(%[[c]])
// CHECK-NEXT: %[[dq_1:.*]] = "tfl.dequantize"(%[[q_1]])
// CHECK-NEXT: %[[v:.*]] = tfl.add %[[dq_1]], %arg2
// CHECK-NEXT: return %[[v:.*]]
}

// -----

// CHECK-LABEL: not_reset_input
func.func @not_reset_input(%arg0: tensor<f32>) -> (tensor<!quant.uniform<i16:f32, 1.0>>) {
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<!quant.uniform<i16:f32, 1.0>>} : (tensor<f32>) -> tensor<!quant.uniform<i16:f32, 1.0>>
  func.return %0: tensor<!quant.uniform<i16:f32, 1.0>>

// CHECK-NEXT: %[[q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<!quant.uniform<i16:f32, 1.000000e+00>>}>
// CHECK-NEXT: return %[[q]]
}

// -----

// CHECK-LABEL: dequantize_and_quantize
func.func @dequantize_and_quantize() -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>> {
  %cst = "tfl.pseudo_qconst"() {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>, value = dense<-1> : tensor<2x2xi8>} : () -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  %0 = "tfl.dequantize"(%cst) : (tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 7.8431372549019615E-4:128>>

// CHECK:  %0 = "tfl.pseudo_qconst"()
// CHECK:  return %0
}

// -----

// CHECK-LABEL: QuantizeAveragePool2D
func.func @QuantizeAveragePool2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.average_pool_2d"(%0) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %1 : tensor<1x1x1x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.average_pool_2d"(%0)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: QuantizeMaximum
func.func @QuantizeMaximum(tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.maximum"(%0, %1) : (tensor<1x6x6x16xf32>, tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %2 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%arg1)
// CHECK: %2 = "tfl.maximum"(%0, %1)
// CHECK: %3 = "tfl.quantize"(%2)
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: return %4 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeMinimum
func.func @QuantizeMinimum(tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 0.1>>) -> tensor<1x6x6x16xf32>
  %2 = "tfl.minimum"(%0, %1) : (tensor<1x6x6x16xf32>, tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  func.return %2 : tensor<1x6x6x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.dequantize"(%arg1)
// CHECK: %2 = "tfl.minimum"(%0, %1)
// CHECK: %3 = "tfl.quantize"(%2)
// CHECK: %4 = "tfl.dequantize"(%3)
// CHECK: return %4 : tensor<1x6x6x16xf32>
}

// -----

// CHECK-LABEL: QuantizeSlice
func.func @QuantizeSlice(tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32> {
^bb0(%arg0: tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x3x5x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x3x5xf32>
  %1 = "tfl.slice"(%0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %1 : tensor<?x3x5xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.slice"(%0, %arg1, %arg2)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<?x3x5xf32>
}

// -----

// CHECK-LABEL: QuantizeStridedSlice
func.func @QuantizeStridedSlice(tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32> {
^bb0(%arg0: tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>) -> tensor<12x2x2x5xf32>
  %1 = "tfl.strided_slice"(%0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  func.return %1 : tensor<1x2x2x5xf32>

// CHECK: %[[dq:.*]] = "tfl.dequantize"(%arg0)
// CHECK: %[[strided_slice:.*]] = "tfl.strided_slice"(%[[dq]], %arg1, %arg2, %arg3)
// CHECK: %[[q:.*]] = "tfl.quantize"(%[[strided_slice]])
// CHECK: %[[dq1:.*]] = "tfl.dequantize"(%[[q]])
// CHECK: return %[[dq1]]
}

// -----

// CHECK-LABEL: QuantizePad
func.func @QuantizePad(tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, tensor<3x2xi32>) -> tensor<?xf32> {
^bb0(%arg0: tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<3x2xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x1x3xf32>
  %1 = "tfl.pad"(%0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.pad"(%0, %arg1)
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3 : tensor<?xf32>
}

// -----

// CHECK-LABEL: QuantizePad2
// only the second tfl.pad has sufficient quantization information.
func.func @QuantizePad2(tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, tensor<2x1x3xf32>, tensor<3x2xi32>) -> (tensor<?xf32>, tensor<?xf32>) {
^bb0(%arg0: tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<2x1x3xf32>, %arg2: tensor<3x2xi32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x1x3xf32>
  %1 = "tfl.pad"(%arg1, %arg2) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  %2 = "tfl.pad"(%0, %arg2) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  func.return %1, %2 : tensor<?xf32>, tensor<?xf32>

// CHECK: %[[dq:.*]] = "tfl.dequantize"(%arg0)
// CHECK: %[[pad1:.*]] = "tfl.pad"(%arg1, %arg2)
// CHECK: %[[pad2:.*]] = "tfl.pad"(%[[dq]], %arg2)
// CHECK: %[[q2:.*]] = "tfl.quantize"(%[[pad2]])
// CHECK: %[[dq2:.*]] = "tfl.dequantize"(%[[q2]])
}

// -----

// CHECK-LABEL: QuantizeReshape2D
func.func @QuantizeReshape2D(tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x6x6x16x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x6x6x16xf32>
  %1 = "tfl.reshape"(%0, %cst) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  func.return %1 : tensor<1x36x16xf32>

// CHECK: %0 = "tfl.dequantize"(%arg0)
// CHECK: %1 = "tfl.reshape"(%0, %{{.*}})
// CHECK: %2 = "tfl.quantize"(%1)
// CHECK: %3 = "tfl.dequantize"(%2)
// CHECK: return %3
}

// -----

// CHECK-LABEL: NotQuantizeConcatConstantOperand
func.func @NotQuantizeConcatConstantOperand(%arg0: tensor<1x2xf32>) -> tensor<2x2xf32> {
  %0 = arith.constant dense<1.0> : tensor<1x2xf32>
  %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK-NEXT: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<1x2xf32>
// CHECK-NEXT: %[[cc:.*]] = "tfl.concatenation"(%arg0, %[[cst]])
// CHECK-NEXT: return %[[cc]]
}

// -----

// CHECK-LABEL: QuantizeConcatOperand0ToAll
func.func @QuantizeConcatOperand0ToAll(tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x2xf32>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHECK: return %5 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: QuantizeConcatOperand1ToAll
func.func @QuantizeConcatOperand1ToAll(tensor<1x2xf32>, tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<2x2xf32> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>):
  %0 = "tfl.dequantize"(%arg1) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%arg0, %0) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>

// CHECK: %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg1) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%1, %2) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %5 = "tfl.dequantize"(%4) : (tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<2x2xf32>
// CHECK: return %5 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAll
func.func @QuantizeConcatResToAll(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %1 = "tfl.quantize"(%0) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %1 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %4 = "tfl.concatenation"(%3, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %5 = "tfl.quantize"(%4) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %5 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllNoRequantize
func.func @QuantizeConcatResToAllNoRequantize(tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 0.1:128>>) -> tensor<1x2xf32>
  %1 = "tfl.concatenation"(%0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %2 = "tfl.quantize"(%1) {qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
  func.return %2 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<u8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %3 = "tfl.concatenation"(%2, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %4 = "tfl.quantize"(%3) <{qtype = tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
// CHECK: return %4 : tensor<2x2x!quant.uniform<u8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllRequantize
func.func @QuantizeConcatResToAllRequantize(tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2xf32>, %arg1: tensor<1x2xf32>):
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>
  %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>

// CHECK: %0 = "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> {propagated} : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %2 = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: %3 = "tfl.dequantize"(%2) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %4 = "tfl.concatenation"(%3, %1) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %5 = "tfl.quantize"(%4) <{qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: return %5 : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
}

// -----

// CHECK-LABEL: QuantizeConcatResToAllRequantizeArg
func.func @QuantizeConcatResToAllRequantizeArg(tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>, tensor<1x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>, %arg1: tensor<1x2xf32>):
  %1 = "tfl.dequantize"(%arg0) : (tensor<1x2x!quant.uniform<i8:f32, 2.0:128>>) -> tensor<1x2xf32>
  %2 = "tfl.concatenation"(%1, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
  %3 = "tfl.quantize"(%2) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
  func.return %3 : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>

// CHECK: %[[Q1:.*]] =  "tfl.quantize"(%arg1) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> {propagated}
// CHECK: %[[DQ1:.*]] = "tfl.dequantize"(%[[Q1]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[RQ0:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<1x2x!quant.uniform<i8:f32, 2.000000e+00:128>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: %[[DQ0:.*]] = "tfl.dequantize"(%[[RQ0]]) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-01:128>>) -> tensor<1x2xf32>
// CHECK: %[[CONC:.*]] = "tfl.concatenation"(%[[DQ0]], %[[DQ1]]) <{axis = 0 : i32, fused_activation_function = "NONE"}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK: %[[Q:.*]] = "tfl.quantize"(%[[CONC]]) <{qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>}> : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
// CHECK: return %[[Q]] : tensor<2x2x!quant.uniform<i8:f32, 1.000000e-01:128>>
}


// -----

// CHECK-LABEL: NotRequantizeAlreadyQuantizedModel
func.func @NotRequantizeAlreadyQuantizedModel(%arg0: tensor<1x73x73x64x!quant.uniform<u8:f32, 1.0>>, %arg1: tensor<1x147x147x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>> {
  %9 = "tfl.max_pool_2d"(%arg1) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x147x147x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x96x!quant.uniform<u8:f32, 2.0>>
  %10 = "tfl.concatenation"(%arg0, %9) {axis = 3 : i32, fused_activation_function = "NONE"} : (tensor<1x73x73x64x!quant.uniform<u8:f32, 1.0>>, tensor<1x73x73x96x!quant.uniform<u8:f32, 2.0>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>>
  func.return %10 : tensor<1x73x73x160x!quant.uniform<u8:f32, 1.0>>

// CHECK: %[[max:.*]] = "tfl.max_pool_2d"(%arg1) <{filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32}> : (tensor<1x147x147x96x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<1x73x73x96x!quant.uniform<u8:f32, 2.000000e+00>>
// CHECK: %[[cat:.*]] = "tfl.concatenation"(%arg0, %[[max]]) <{axis = 3 : i32, fused_activation_function = "NONE"}> : (tensor<1x73x73x64x!quant.uniform<u8:f32, 1.000000e+00>>, tensor<1x73x73x96x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<1x73x73x160x!quant.uniform<u8:f32, 1.000000e+00>>
// CHECK: return %[[cat]] : tensor<1x73x73x160x!quant.uniform<u8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: QuantizeChain
func.func @QuantizeChain(tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x36x16xf32> {
^bb0(%arg0: tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>):
  %cst = arith.constant dense<-1.23697901> : tensor<32xf32>
  %cst_0 = arith.constant dense<[1, 36, 16]> : tensor<3xi32>
  %2 = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>) -> tensor<1x224x224x3xf32>
  %3 = "tfl.pseudo_qconst"() {qtype = tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>, value = dense<-76> : tensor<32x3x3x3xi8>} : () -> tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>
  %4 = "tfl.dequantize"(%3) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>) -> tensor<32x3x3x3xf32>
  %5 = "tfl.average_pool_2d"(%2) {
      name = "avgpool", filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32
    } : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %6 = "tfl.conv_2d"(%5, %4, %cst) {
      dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32
    } : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32xf32>) -> tensor<1x112x112x32xf32>
  %7 = "tfl.quantize"(%6) {qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>} : (tensor<1x112x112x32xf32>) -> tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>
  %8 = "tfl.dequantize"(%7) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>) -> tensor<1x6x6x16xf32>
  %9 = "tfl.reshape"(%8, %cst_0) : (tensor<1x6x6x16xf32>, tensor<3xi32>) -> tensor<1x36x16xf32>
  %10 = "tfl.softmax"(%9) {beta = 1.000000e+00 : f32} : (tensor<1x36x16xf32>) -> tensor<1x36x16xf32>
  func.return %10 : tensor<1x36x16xf32>

// CHECK-DAG: %[[cst:.*]] = arith.constant dense<-1.23697901> : tensor<32xf32>
// CHECK-DAG: %[[dq:.*]] = "tfl.dequantize"(%arg0) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK-DAG: %[[q_cst:.*]] = "tfl.pseudo_qconst"()
// CHECK-DAG: %[[dq_cst:.*]] = "tfl.dequantize"(%[[q_cst]]) : (tensor<32x3x3x3x!quant.uniform<u8<1:255>:f32, 0.021826678373682216:151>>)
// CHECK: %[[avg_pool:.*]] = "tfl.average_pool_2d"(%[[dq]])
// CHECK: %[[q_avg_pool:.*]] = "tfl.quantize"(%[[avg_pool]]) <{qtype = tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>}> {propagated}
// CHECK: %[[dq_avg_pool:.*]] = "tfl.dequantize"(%[[q_avg_pool]]) : (tensor<1x224x224x3x!quant.uniform<u8:f32, 7.812500e-03:128>>)
// CHECK: %[[conv:.*]] = "tfl.conv_2d"(%[[dq_avg_pool]], %[[dq_cst]], %[[cst]])
// CHECK: %[[q_conv:.*]] = "tfl.quantize"(%[[conv]]) <{qtype = tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>}>
// CHECK: %[[dq_conv:.*]] = "tfl.dequantize"(%[[q_conv]]) : (tensor<1x112x112x32x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %[[reshape:.*]] = "tfl.reshape"(%[[dq_conv]], %{{.*}})
// CHECK: %[[q_reshape:.*]] = "tfl.quantize"(%[[reshape]]) <{qtype = tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>}> {propagated}
// CHECK: %[[dq_reshape:.*]] = "tfl.dequantize"(%[[q_reshape]]) : (tensor<1x36x16x!quant.uniform<u8:f32, 0.023528476789885875>>)
// CHECK: %[[softmax:.*]] = "tfl.softmax"(%[[dq_reshape]])
// CHECK: return %[[softmax]] : tensor<1x36x16xf32>
}

// -----

// CHECK-LABEL: NotQuantizeNoneType
func.func @NotQuantizeNoneType() -> none {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  func.return %cst : none

// CHECK-NEXT:  %[[cst:.*]] = "tfl.no_value"() <{value}> : () -> none
// CHECK-NEXT:  return %[[cst]]
}

// -----

// Make sure constants are duplicataed for all users.
// CHECK-LABEL: QuantizeSharedConstantsMultipleUsers
func.func @QuantizeSharedConstantsMultipleUsers(
    %arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>,
    %arg1: tensor<32x!quant.uniform<u8:f32, 2.0>>,
    %arg2: tensor<32x!quant.uniform<u8:f32, 3.0>>,
    %arg3: tensor<32x!quant.uniform<u8:f32, 4.0>>) -> (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) {
  %cst = arith.constant dense<0.0> : tensor<32xf32>
  %0 = "tfl.dequantize"(%arg0) : (tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32xf32>
  %1 = "tfl.dequantize"(%arg1) : (tensor<32x!quant.uniform<u8:f32, 2.0>>) -> tensor<32xf32>
  %2 = "tfl.dequantize"(%arg2) : (tensor<32x!quant.uniform<u8:f32, 3.0>>) -> tensor<32xf32>
  %3 = "tfl.dequantize"(%arg3) : (tensor<32x!quant.uniform<u8:f32, 4.0>>) -> tensor<32xf32>

  %4 = "tfl.minimum"(%0, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %5 = "tfl.minimum"(%1, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %6 = "tfl.minimum"(%2, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  %7 = "tfl.minimum"(%3, %cst) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
  func.return %4, %5, %6, %7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>

// CHECK-DAG: %[[cst1:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 1.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst2:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 2.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst3:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 3.000000e+00>>) -> tensor<32xf32>
// CHECK-DAG: %[[cst4:.*]] = "tfl.dequantize"(%{{.*}}) : (tensor<32x!quant.uniform<u8:f32, 4.000000e+00>>) -> tensor<32xf32>
// CHECK-NOT: BLOCK_DAG
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst1]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst2]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst3]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
// CHECK-DAG: "tfl.minimum"(%{{.*}}, %[[cst4]]) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
}

// -----

// CHECK-LABEL: TransposePerTensorQuantizationPropagation
func.func @TransposePerTensorQuantizationPropagation() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32, 1.113490e-03>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

  // CHECK: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
  // CHECK: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32
  // CHECK: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32
  // CHECK: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  // CHECK: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i8<-127:127>:f32
  // CHECK: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32
  // CHECK: return %[[dqtw]] : tensor<2x5xf32>
}

// CHECK-LABEL: TransposePerChannelNewQuantDim
func.func @TransposePerChannelNewQuantDim() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

// CHECK: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
// CHECK: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i8<-127:127>:f32:0
// CHECK: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i8<-127:127>:f32:0
// CHECK: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
// CHECK: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i8<-127:127>:f32:1
// CHECK: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i8<-127:127>:f32:1
// CHECK: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: ReshapePerChannelNewQuantDim
func.func @ReshapePerChannelNewQuantDim() -> tensor<24x5xf32> {
  %cst = arith.constant dense<1.0> : tensor<1x2x3x4x5xf32>
  %cst_1 = arith.constant dense<[24, 5]> : tensor<2xi32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>} : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>
  %dq = "tfl.dequantize"(%q) : (tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {0.2345, 0.2345, 0.2345, 0.2345, 0.2345}>>) -> tensor<1x2x3x4x5xf32>
  %0 = "tfl.reshape"(%dq, %cst_1) : (tensor<1x2x3x4x5xf32>, tensor<2xi32>) -> tensor<24x5xf32>
  func.return %0 : tensor<24x5xf32>

// CHECK: %cst = arith.constant dense<1.000000e+00> : tensor<1x2x3x4x5xf32>
// CHECK: %cst_0 = arith.constant dense<[24, 5]> : tensor<2xi32>
// CHECK: %0 = "tfl.quantize"(%cst) <{qtype = tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>}> : (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>
// CHECK: %1 = "tfl.dequantize"(%0) : (tensor<1x2x3x4x5x!quant.uniform<i4:f32:4, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>) -> tensor<1x2x3x4x5xf32>
// CHECK: %2 = "tfl.reshape"(%1, %cst_0) : (tensor<1x2x3x4x5xf32>, tensor<2xi32>) -> tensor<24x5xf32>
// CHECK: %3 = "tfl.quantize"(%2) <{qtype = tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>}> {propagated} : (tensor<24x5xf32>) -> tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>
// CHECK: %4 = "tfl.dequantize"(%3) : (tensor<24x5x!quant.uniform<i4:f32:1, {2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01,2.345000e-01}>>) -> tensor<24x5xf32>
// CHECK: return %4 : tensor<24x5xf32>
}

// -----

// CHECK-LABEL: TransposePerChannelNewQuantDim_int4
func.func @TransposePerChannelNewQuantDim_int4() -> tensor<2x5xf32> {
  %perm = arith.constant dense<[1, 0]> : tensor<2xi32>
  %cst = arith.constant dense<1.0> : tensor<5x2xf32>
  %q = "tfl.quantize"(%cst) {qtype = tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>} : (tensor<5x2xf32>) -> tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>
  %dq = "tfl.dequantize"(%q) : (tensor<5x2x!quant.uniform<i4<-7:7>:f32:0, {1.0,2.0,3.0,4.0,5.0}>>) -> tensor<5x2xf32>
  %t = "tfl.transpose"(%dq, %perm) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  func.return %t : tensor<2x5xf32>

// CHECK: %[[perm:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[w:.*]] = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
// CHECK: %[[qw:.*]] = "tfl.quantize"(%[[w]]) <{qtype = tensor<5x2x!quant.uniform<i4<-7:7>:f32:0
// CHECK: %[[dqw:.*]] = "tfl.dequantize"(%[[qw]]) : (tensor<5x2x!quant.uniform<i4<-7:7>:f32:0
// CHECK: %[[tp:.*]] = "tfl.transpose"(%[[dqw]], %[[perm]]) : (tensor<5x2xf32>, tensor<2xi32>) -> tensor<2x5xf32>
// CHECK: %[[qtw:.*]] = "tfl.quantize"(%[[tp]]) <{qtype = tensor<2x5x!quant.uniform<i4<-7:7>:f32:1
// CHECK: %[[dqtw:.*]] = "tfl.dequantize"(%[[qtw]]) : (tensor<2x5x!quant.uniform<i4<-7:7>:f32:1
// CHECK: return %[[dqtw]] : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: DQQToRequantize
func.func @DQQToRequantize(%arg0: tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>) {
    %0 = "tfl.dequantize"(%arg0) : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320xf32>
    %1 = "tfl.quantize"(%0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320xf32>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
    return %1 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>

// CHECK: %0 = "tfl.quantize"(%arg0) <{qtype = tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>}> : (tensor<1x128x128x320x!quant.uniform<i8:f32, 0.17072822153568268:6>>) -> tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
// CHECK: return %0 : tensor<1x128x128x320x!quant.uniform<i8:f32, 0.1043805405497551:-6>>
}
