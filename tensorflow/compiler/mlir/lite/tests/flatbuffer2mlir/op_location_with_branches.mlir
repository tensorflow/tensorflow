// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --mlir-print-debuginfo - -o - | FileCheck %s
// Verifies imported op locations carry TFLite subgraph/op ids and fused activation.

func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK:      %[[ADD:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<1xf32> loc("tflite.subgraph=0.op=0:ADD_fused_RELU6")
  // CHECK-NEXT: %[[LESS:.*]] = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc("tflite.subgraph=0.op=1:LESS")
  // CHECK-NEXT: %[[IF:.*]] = "tf.If"(%[[LESS]], %[[ADD]], %arg1) <{else_branch = @cond_false, is_stateless = false, then_branch = @cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc("tflite.subgraph=0.op=2:IF")
  // CHECK-NEXT: return %[[IF]] : tensor<1xf32>{{.*}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<1xf32>
  %1 = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %2 = "tf.If"(%1, %0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %2 : tensor<1xf32>
}

func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:      func.func @cond_true
  // CHECK-NEXT: %[[ADD_TRUE:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc("tflite.subgraph=1.op=0:ADD")
  // CHECK-NEXT: return %[[ADD_TRUE]] : tensor<*xf32>{{.*}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:      func.func @cond_false
  // CHECK-NEXT: %[[MUL_FALSE:.*]] = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc("tflite.subgraph=2.op=0:MUL")
  // CHECK-NEXT: return %[[MUL_FALSE]] : tensor<*xf32>{{.*}}
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
