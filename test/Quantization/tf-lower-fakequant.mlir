// RUN: mlir-opt %s -split-input-file -quant-lower-tf | FileCheck %s --dump-input=fail

// -----
// Verifies a quint8 asymmetric 0..1 range.
// CHECK-LABEL: fakeQuantArgs_Quint8_0_1
func @fakeQuantArgs_Quint8_0_1(tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
^bb0(%arg0: tensor<8x4x3xf32>):
  // CHECK: %0 = "quant.qbarrier"(%arg0) : (tensor<8x4x3xf32>)
  // CHECK-SAME: -> tensor<8x4x3x!quant<"uniform[u8:f32]{0.0039215686274509803}">>
  // CHECK-NEXT: %1 = "quant.dbarrier"(%0) : (tensor<8x4x3x!quant<"uniform[u8:f32]{0.0039215686274509803}">>)
  // CHECK-SAME: -> tensor<8x4x3xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min: 0.0, max: 1.0, num_bits: 8
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// Verifies a quint8 asymmetric 0..1 range (with narrow_range = true).
// CHECK_LABEL: fakeQuantArgs_Quint8_NarrowRange
func @fakeQuantArgs_Quint8_NarrowRange(tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
^bb0(%arg0: tensor<8x4x3xf32>):
  // CHECK: %0 = "quant.qbarrier"(%arg0) : (tensor<8x4x3xf32>)
  // CHECK-SAME: -> tensor<8x4x3x!quant<"uniform[u8(1:255):f32]{0.003937007874015748:1}">>
  // CHECK-NEXT: %1 = "quant.dbarrier"(%0) : (tensor<8x4x3x!quant<"uniform[u8(1:255):f32]{0.003937007874015748:1}">>)
  // CHECK-SAME: -> tensor<8x4x3xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min: 0.0, max: 1.0, num_bits: 8, narrow_range: true
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// Verifies a quint8 symmetric range of -1..127/128.
// CHECK_LABEL: fakeQuantArgs_Quint8_SymmetricRange
func @fakeQuantArgs_Quint8_SymmetricRange(tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
^bb0(%arg0: tensor<8x4x3xf32>):
  // CHECK: %0 = "quant.qbarrier"(%arg0) : (tensor<8x4x3xf32>)
  // CHECK-SAME: -> tensor<8x4x3x!quant<"uniform[u8:f32]{7.812500e-03:128}">>
  // CHECK-NEXT: %1 = "quant.dbarrier"(%0) : (tensor<8x4x3x!quant<"uniform[u8:f32]{7.812500e-03:128}">>)
  // CHECK-SAME: -> tensor<8x4x3xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min: -1.0, max: 0.9921875, num_bits: 8, narrow_range: false
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// Verifies a commonly used -1..1 symmetric 16bit range with a zero point of
// 0 and range -1.0 .. 32767/32768.
// CHECK-LABEL: fakeQuantArgs_Qint16_Symmetric
func @fakeQuantArgs_Qint16_Symmetric(tensor<8x4x3xf32>) -> tensor<8x4x3xf32> {
^bb0(%arg0: tensor<8x4x3xf32>):
  // CHECK: %0 = "quant.qbarrier"(%arg0) : (tensor<8x4x3xf32>)
  // CHECK-SAME: -> tensor<8x4x3x!quant<"uniform[i16:f32]{3.05175781185626E-5}">>
  // CHECK-NEXT: %1 = "quant.dbarrier"(%0) : (tensor<8x4x3x!quant<"uniform[i16:f32]{3.05175781185626E-5}">>)
  // CHECK-SAME: -> tensor<8x4x3xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min: -1.0, max: 0.999969482, num_bits: 16
  } : (tensor<8x4x3xf32>) -> tensor<8x4x3xf32>
  return %0 : tensor<8x4x3xf32>
}

// -----
// Verify that lowering to barriers of unranked tensors functions.
// CHECK-LABEL: fakeQuantArgs_UnrankedTensor
func @fakeQuantArgs_UnrankedTensor(tensor<f32>) -> tensor<f32> {
^bb0(%arg0: tensor<f32>):
  // CHECK: %0 = "quant.qbarrier"(%arg0) : (tensor<f32>)
  // CHECK-SAME: -> tensor<!quant<"uniform[u8:f32]{0.0039215686274509803}">>
  // CHECK-NEXT: %1 = "quant.dbarrier"(%0) : (tensor<!quant<"uniform[u8:f32]{0.0039215686274509803}">>)
  // CHECK-SAME: -> tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min: 0.0, max: 1.0, num_bits: 8
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
