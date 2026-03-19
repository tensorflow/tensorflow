// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

func.func @main(%arg0 : tensor<!tf_type.variant<tensor<2xi32>>>, %arg1: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>> {
  func.return %arg0 : tensor<!tf_type.variant<tensor<2xi32>>>
}

// CHECK:          func.func @main(%[[ARG0:.*]]: tensor<!tf_type.variant<tensor<2xi32>>>, %[[ARG1:.*]]: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>>
// CHECK-NEXT:       return %[[ARG0]] : tensor<!tf_type.variant<tensor<2xi32>>>
