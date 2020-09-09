// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

module attributes {
  tfl.metadata = {key1 = "value1", key2 = "value2"}
} {
  func @main(tensor<3x2xi32>) -> tensor<3x2xi32>
    attributes {tf.entry_function = {inputs = "input", outputs = "SameNameAsOutput"}} {
  ^bb0(%arg0: tensor<3x2xi32>):
    %0 = "tfl.pseudo_const" () {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
    %1 = "tfl.sub" (%arg0, %0) {fused_activation_function = "NONE"} : (tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
    return %1 : tensor<3x2xi32>
  }
}

// CHECK:      buffers: [ {
// CHECK:      }, {
// CHECK:      }, {
// CHECK:      }, {
// CHECK:      }, {
// CHECK-NEXT:   data: [ 118, 97, 108, 117, 101, 49 ]
// CHECK-NEXT: }, {
// CHECK-NEXT:   data: [ 118, 97, 108, 117, 101, 50 ]
// CHECK-NEXT: }, {
// CHECK-NEXT:   data: [ 49, 46, 54, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT: } ],
// CHECK-NEXT: metadata: [ {
// CHECK-NEXT:   name: "key1",
// CHECK-NEXT:   buffer: 4
// CHECK-NEXT: }, {
// CHECK-NEXT:   name: "key2",
// CHECK-NEXT:   buffer: 5
// CHECK-NEXT: }, {
// CHECK-NEXT:   name: "min_runtime_version",
// CHECK-NEXT:   buffer: 6
// CHECK-NEXT: } ]
// CHECK-NEXT: }
