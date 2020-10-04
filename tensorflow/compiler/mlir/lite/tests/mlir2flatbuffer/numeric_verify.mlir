// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

// CHECK:      {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:    builtin_code: CUSTOM,
// CHECK-NEXT:    custom_code: "NumericVerify"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: UINT8,
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-NEXT:        scale: [ 0.1 ],
// CHECK-NEXT:        zero_point: [ 0 ]
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 0 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 1, 0 ],
// CHECK-NEXT:      outputs: [  ],
// CHECK-NEXT:      custom_options: [ 205, 204, 204, 61 ]
// CHECK-NEXT:    } ],
// CHECK-NEXT:    name: "main"
// CHECK-NEXT:  } ],
// CHECK-NEXT:  description: "MLIR Converted.",
// CHECK-NEXT:  buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:  name: "min_runtime_version",
// CHECK-NEXT:  buffer: 3
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

func @main(%arg0: tensor<4xf32>, %arg1: tensor<4x!quant.uniform<u8:f32, 0.1>>) -> tensor<4xf32> {
  "tfl.NumericVerify"(%arg1, %arg0) {tolerance = 0.1 : f32} : (tensor<4x!quant.uniform<u8:f32, 0.1>>, tensor<4xf32>) -> ()
  return %arg0 : tensor<4xf32>
}
