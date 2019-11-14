// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Check to see if function references in while loops are preserved
func @main(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// TODO(b/138222071) Expect first output to be a scalar
// CHECK:   %{{.*}}:2 = "tf.While"(%{{.*}}, %{{.*}}) {body = @body, cond = @cond, is_stateless = false} : (tensor<i32>, tensor<1xf32>) -> (tensor<*xi32>, tensor<1xf32>)

  // While %arg0 is greater than zero, element wise add %arg1 with itself.
  %0:2 = "tf.While"(%arg0, %arg1) {
    cond = @cond, body = @body, is_stateless = false
  } : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  return %0#1 : tensor<1xf32>
}

func @cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %0 = "std.constant" () {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
  %1 = "tfl.greater"(%arg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

func @body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %0 = "std.constant" () {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
  %1 = "tfl.sub"(%arg0, %0) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %2 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %1, %2 : tensor<*xi32>, tensor<*xf32>
}
