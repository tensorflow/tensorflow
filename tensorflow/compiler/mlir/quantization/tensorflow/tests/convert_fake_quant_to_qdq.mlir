// RUN: tf-quant-opt %s -quant-convert-fake-quant-to-qdq | FileCheck %s

func.func @fakeQuantArgs(%arg0: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min = -0.1 : f32, max = 0.2 : f32, num_bits = 8
  } : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}
// CHECK: func @fakeQuantArgs
// CHECK-NEXT: %[[q:.*]] = "quantfork.qcast"(%arg0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8x!quant.uniform<i8:f32, 0.0011764706057660721:-43>>
// CHECK-NEXT: %[[dq:.*]] = "quantfork.dcast"(%[[q]])
// CHECK-NEXT: return %[[dq]]

func.func @doNotHandleNonEightBitFakeQuant(%arg0: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {
    min = -0.1 : f32, max = 0.2 : f32, num_bits = 16
  } : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}
// CHECK: func @doNotHandleNonEightBitFakeQuant
// CHECK: tf.FakeQuantWithMinMaxArgs
// CHECK-NOT: "quantfork.qcast"

func.func @fakeQuantVars(%arg0: tensor<3xf32>, %arg1: tensor<4x3xf32>) -> (tensor<3xf32>, tensor<4x3xf32>) {
  %cst = "tf.Const"() {value = dense<-0.950868546> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<9.951540e-01> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<[-0.5, -0.4, -0.7]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.5, 0.6, 0.3]> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %cst, %cst_0) {
    device = "", narrow_range = false, num_bits = 8 : i64
  } : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  %1 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg1, %cst_1, %cst_2) {
    device = "", narrow_range = true, num_bits = 8 : i64
  } : (tensor<4x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<4x3xf32>
  func.return %0, %1 : tensor<3xf32>, tensor<4x3xf32>
}

// CHECK: %[[q1:.*]] = "quantfork.qcast"(%arg0)
// CHECK-SAME: tensor<3x!quant.uniform<i8:f32, 0.0076314610593459188:-3>>
// CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
// CHECK: %[[q2:.*]] = "quantfork.qcast"(%arg1)
// CHECK-SAME: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {0.003937007874015748,0.0039370079913477263:-25,0.003937007874015748:51}>>
// CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
// CHECK: return %[[dq1]], %[[dq2]]
