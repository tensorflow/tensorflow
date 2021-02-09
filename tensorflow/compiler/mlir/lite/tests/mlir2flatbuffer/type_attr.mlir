// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -emit-builtin-tflite-ops=false -o - | flatbuffer_to_string - | FileCheck %s

// CHECK: {
// CHECK:   version: 3,
// CHECK:   operator_codes: [ {
// CHECK:     deprecated_builtin_code: 32,
// CHECK:     custom_code: "SomeOperation",
// CHECK:     builtin_code: CUSTOM
// CHECK:   } ],
// CHECK:   subgraphs: [ {
// CHECK:     tensors: [ {
// CHECK:       shape: [  ],
// CHECK:       type: INT32,
// CHECK:       buffer: 1,
// CHECK:       name: "tf.SomeOperation",
// CHECK:       quantization: {
// CHECK-EMPTY
// CHECK:       }
// CHECK:     } ],
// CHECK:     inputs: [  ],
// CHECK:     outputs: [ 0 ],
// CHECK:     operators: [ {
// CHECK:       inputs: [  ],
// CHECK:       outputs: [ 0 ],
// CHECK:       custom_options: [ 100, 116, 121, 112, 101, 0, 1, 7, 1, 1, 1, 2, 4, 2, 36, 1 ]
// CHECK:     } ],
// CHECK:     name: "main"
// CHECK:   } ],
// CHECK:   description: "MLIR Converted.",
// CHECK:   buffers: [ {
// CHECK-EMPTY
// CHECK:   }, {
// CHECK-EMPTY
// CHECK:   } ]
// CHECK: }

func @main() -> tensor<*xi32> {
	// Tests that the below type attribute is convertible into the corresponding custom option in flatbuffer.
  %0 = "tf.SomeOperation"() {dtype = i32 } : () -> tensor<*xi32>
  return %0 : tensor<*xi32>
}
