// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string -

func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %cst = "tfl.pseudo_const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<?xi32>
  %0 = "tfl.add"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<?xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}


// CHECK:    tensors: [ {
// CHECK-NEXT:      shape: [ 2 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "tfl.pseudo_const",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:
// CHECK-NEXT:      }

// CHECK:   buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:   }, {
// CHECK-NEXT:     data: [ 1, 0, 0, 0, 2, 0, 0, 0 ]
// CHECK-NEXT:   }, {
