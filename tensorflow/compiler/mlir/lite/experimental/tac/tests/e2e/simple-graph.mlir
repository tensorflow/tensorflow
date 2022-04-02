// RUN: tac-translate -input-mlir -output-mlir -device-specs=GPU %s -o - 2>&1 | FileCheck %s

module {
func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> attributes {tf.entry_function = {inputs = "input0,input1,input2,input3", outputs = "output"}} {
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%0, %arg2) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.add"(%arg0, %arg3) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.pack"(%1, %2) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  func.return %3 : tensor<2x1xf32>
}

// CHECK:  [[VAL_0:%.*]] = "tfl.reshape"(%1, %cst) {tac.device = "GPU",  tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:  [[VAL_1:%.*]] = "tfl.reshape"(%2, %cst) {tac.device = "GPU",  tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:  [[VAL_2:%.*]] = "tfl.concatenation"([[VAL_0]], [[VAL_1]]) {axis = 3 : i32, fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x2xf32>
// CHECK:  [[VAL_3:%.*]] = "tfl.reshape"([[VAL_2]], %cst_0) : (tensor<1x1x1x2xf32>, tensor<2xi32>) -> tensor<2x1xf32>

}
