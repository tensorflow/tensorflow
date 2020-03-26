// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate -tflite-flatbuffer-to-mlir - -o - | FileCheck --check-prefix=IMPORT %s

func @main(tensor<4xf32>) -> tensor<4xf32> {
^bb0(%arg0: tensor<4xf32>):
// CHECK: {
// CHECK-NEXT:   version: 3,
// CHECK-NEXT:     operator_codes: [ {
// CHECK-NEXT:       builtin_code: FAKE_QUANT,
// CHECK-NEXT:       version: 1
// CHECK-NEXT:     } ],
// CHECK-NEXT:     subgraphs: [ {
// CHECK-NEXT:       tensors: [ {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 1,
// CHECK-NEXT:         name: "arg0",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       }, {
// CHECK-NEXT:         shape: [ 4 ],
// CHECK-NEXT:         buffer: 2,
// CHECK-NEXT:         name: "tfl.fake_quant",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         }
// CHECK-NEXT:       } ],
// CHECK-NEXT:       inputs: [ 0 ],
// CHECK-NEXT:       outputs: [ 1 ],
// CHECK-NEXT:       operators: [ {
// CHECK-NEXT:         inputs: [ 0 ],
// CHECK-NEXT:         outputs: [ 1 ],
// CHECK-NEXT:         builtin_options_type: FakeQuantOptions,
// CHECK-NEXT:         builtin_options: {
// CHECK-NEXT:           min: 0.3,
// CHECK-NEXT:           max: 1.4,
// CHECK-NEXT:           num_bits: 6
// CHECK-NEXT:         }
// CHECK-NEXT:       } ],
// CHECK-NEXT:       name: "main"
// CHECK-NEXT:     } ],
// CHECK-NEXT:     description: "MLIR Converted.",
// CHECK-NEXT:     buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-NEXT:       data: [ 49, 46, 53, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:     } ],
// CHECK-NEXT:     metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 3
// CHECK-NEXT:     } ]
// CHECK-NEXT:   }

// IMPORT: "tfl.fake_quant"(%arg0) {max = 1.400000e+00 : f32, min = 3.000000e-01 : f32, narrow_range = false, num_bits = 6 : i32}

  %0 = "tfl.fake_quant"(%arg0) {num_bits = 6 : i32, narrow_range = false, min = 0.3:f32, max = 1.4:f32} : (tensor<4 x f32>) -> tensor<4 x f32>
  return %0 : tensor<4xf32>
}
