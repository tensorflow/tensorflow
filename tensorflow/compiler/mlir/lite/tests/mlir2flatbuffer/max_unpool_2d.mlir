// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir -o - | FileCheck --check-prefix=MLIR %s

func @main(%arg0: tensor<1x8x8x128xf32>, %arg1: tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32> {

// CHECK:  {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:      builtin_code: CUSTOM,
// CHECK-NEXT:      custom_code: "MaxUnpooling2D"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    subgraphs: [ {
// CHECK-NEXT:      tensors: [ {
// CHECK-NEXT:        shape: [ 1, 8, 8, 128 ],
// CHECK-NEXT:        buffer: 1,
// CHECK-NEXT:        name: "arg0",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      }, {
// CHECK-NEXT:        shape: [ 1, 8, 8, 128 ],
// CHECK-NEXT:        buffer: 2,
// CHECK-NEXT:        name: "arg1",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 8, 8, 128 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "tfl.max_unpooling_2d",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 2 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      custom_options: [ 1, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
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
// CHECK-EMPTY:
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:  name: "min_runtime_version",
// CHECK-NEXT:  buffer: 4
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

// MLIR-LABEL:  func @main(%arg0: tensor<1x8x8x128xf32>, %arg1: tensor<1x8x8x128xf32>)
// MLIR-SAME:    -> tensor<1x8x8x128xf32>
// MLIR:         %0 = "tfl.max_unpooling_2d"(%arg0, %arg1)
// MLIR-SAME:     {filter_h = 1 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 4 : i32, stride_w = 2 : i32}
// MLIR-SAME:     (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32>
// MLIR-NEXT:    return %0 : tensor<1x8x8x128xf32>

  %0 = "tfl.max_unpooling_2d"(%arg0, %arg1) {filter_h = 1 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 4 : i32, stride_w = 2 : i32} : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> (tensor<1x8x8x128xf32>)
  return %0 : tensor<1x8x8x128xf32>
}
