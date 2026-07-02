// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --mlir-print-debuginfo - -o - | FileCheck %s
// Verifies imported op locations carry TFLite subgraph/op ids and fused activation.

// CHECK-DAG: #[[STDIN_LOC:.*]] = loc("<stdin>":0:0)

func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK:      %[[ADD:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<1xf32> loc(#[[ADD_LOC:.*]])
  // CHECK-NEXT: %[[LESS:.*]] = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1> loc(#[[LESS_LOC:.*]])
  // CHECK-NEXT: %[[IF:.*]] = "tf.If"(%[[LESS]], %[[ADD]], %arg1) <{else_branch = @cond_false, is_stateless = false, then_branch = @cond_true}> : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc(#[[IF_LOC:.*]])
  // CHECK-NEXT: return %[[IF]] : tensor<1xf32>{{.*}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<1xf32>
  %1 = tfl.less(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
  %2 = "tf.If"(%1, %0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %2 : tensor<1xf32>
}

func.func @cond_true(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:      func.func{{.*}}@cond_true
  // CHECK-NEXT: %[[ADD_TRUE:.*]] = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#[[ADD_TRUE_LOC:.*]])
  // CHECK-NEXT: return %[[ADD_TRUE]] : tensor<*xf32>{{.*}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @cond_false(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:      func.func{{.*}}@cond_false
  // CHECK-NEXT: %[[MUL_FALSE:.*]] = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32> loc(#[[MUL_FALSE_LOC:.*]])
  // CHECK-NEXT: return %[[MUL_FALSE]] : tensor<*xf32>{{.*}}
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-DAG: #[[ADD_TENSOR:.*]] = loc("tfl.add"(#[[STDIN_LOC]]))
// CHECK-DAG: #[[ADD_LOC]] = loc("tflite.subgraph=0.op=0:ADD_fused_RELU6"(#[[ADD_TENSOR]]))
// CHECK-DAG: #[[LESS_TENSOR:.*]] = loc("tfl.less"(#[[STDIN_LOC]]))
// CHECK-DAG: #[[LESS_LOC]] = loc("tflite.subgraph=0.op=1:LESS"(#[[LESS_TENSOR]]))
// CHECK-DAG: #[[IF_TENSOR:.*]] = loc("tf.If"(#[[STDIN_LOC]]))
// CHECK-DAG: #[[IF_LOC]] = loc("tflite.subgraph=0.op=2:IF"(#[[IF_TENSOR]]))
// CHECK-DAG: #[[ADD_TRUE_TENSOR:.*]] = loc("tfl.add1"(#[[STDIN_LOC]]))
// CHECK-DAG: #[[ADD_TRUE_LOC]] = loc("tflite.subgraph=1.op=0:ADD"(#[[ADD_TRUE_TENSOR]]))
// CHECK-DAG: #[[MUL_FALSE_TENSOR:.*]] = loc("tfl.mul"(#[[STDIN_LOC]]))
// CHECK-DAG: #[[MUL_FALSE_LOC]] = loc("tflite.subgraph=2.op=0:MUL"(#[[MUL_FALSE_TENSOR]]))
