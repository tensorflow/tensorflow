// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(%arg0: tensor<*x!quant.uniform<u16:f32, 2.0:37>>) -> tensor<*x!quant.uniform<u16:f32, 2.0:37>> {
// CHECK:     {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [  ],
// CHECK-NEXT:      type: UINT16,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 2.0 ],
// CHECK-NEXT:        zero_point: [ 37 ]
// CHECK:           }
// CHECK-NEXT:    } ],
  return %arg0 : tensor<*x!quant.uniform<u16:f32, 2.0:37>>
}
