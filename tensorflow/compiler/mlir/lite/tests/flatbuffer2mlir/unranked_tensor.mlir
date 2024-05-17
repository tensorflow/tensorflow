// RUN: tf-opt --tfl-legalize-tf-while %s -o - | flatbuffer_translate -mlir-to-tflite-flatbuffer - -o -  | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK-LABEL: main
func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:     %{{.*}} = tfl.add %{{.*}}, %{{.*}} {fused_activation_function = "NONE"} : tensor<*xf32>
  // CHECK:     return %{{.*}} : tensor<*xf32>

  %0 = tfl.add(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}