// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Ensure lstm roundtrip exactly

func @main(%arg0: tensor<4 x f32>, %arg1: tensor<4 x f32>, %arg2: tensor<4 x f32>, %arg3: tensor<4 x f32>, %arg4: tensor<4 x f32>, %arg5: tensor<4 x f32>, %arg6: tensor<4 x f32>, %arg7: tensor<4 x f32>, %arg8: tensor<4 x f32>, %arg9: tensor<4 x f32>, %arg10: tensor<4 x f32>, %arg11: tensor<4 x f32>, %arg12: tensor<4 x f32>, %arg13: tensor<4 x f32>, %arg14: tensor<4 x f32>, %arg15: tensor<4 x f32>, %arg16: tensor<4 x f32>, %arg17: tensor<4 x f32>, %arg18: tensor<4 x f32>, %arg19: tensor<4 x f32>, %arg20: tensor<4 x f32>, %arg21: tensor<4 x f32>) -> tensor<4 x f32> {
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {fused_activation_function = "NONE", kernel_type = "FULL"} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %24 : tensor<4xf32>
// CHECK-LABEL: main
// seperate lines since there is no region for this op. third_party/tensorflow/compiler/mlir/lite/ir/tfl_ops.td: 3252
// CHECK: %[[RES0:.*]] = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg22, %arg23, %arg18, %arg19, %arg20, %arg21) ( {
// CHECK:  }) {cell_clip = 0.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = "FULL", proj_clip = 0.000000e+00 : f32} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK: return %[[RES0]]

}
