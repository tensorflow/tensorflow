// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck --dump-input-on-failure %s
// Check a few basic properties of the import-export,
// including constants retaining their shape
// and the module including the TFLite version.

func @main(tensor<3x2xi32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>):
  // CHECK: module attributes {tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}

  // CHECK:          %{{.*}} = "tfl.pseudo_const"() {value = dense<{{\[\[1, 2\], \[3, 4\], \[5, 6\]\]}}> : tensor<3x2xi32>}
  // CHECK-NEXT:     [[SUB:%.*]] = tfl.sub %{{.*}}, %{{.*}} {fused_activation_function = "RELU6"} : tensor<3x2xi32>
  // CHECK-NEXT:     [[SCALAR:%.*]] = "tfl.pseudo_const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT:     [[ADD:%.*]] = "tfl.add"([[SCALAR]], [[SUB]]) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  // CHECK-NEXT:     return [[ADD]] : tensor<3x2xi32>

  %0 = "tfl.pseudo_input" (%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi32> loc("Input")
  %1 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32> loc("Const")
  %2 = "tfl.sub" (%0, %1) {fused_activation_function = "RELU6"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("sub")
  %3 = "std.constant" () {value = dense<10> : tensor<i32>} : () -> tensor<i32> loc("Const2")
  %4 = "tfl.add" (%3, %2) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<3x2xi32>) -> tensor<3x2xi32> loc("add")
  return %4 : tensor<3x2xi32>
}
