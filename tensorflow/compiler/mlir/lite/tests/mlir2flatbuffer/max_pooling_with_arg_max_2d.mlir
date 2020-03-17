// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -o - | flatbuffer_to_string - | FileCheck --dump-input-on-failure %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir -o - | FileCheck --check-prefix=MLIR %s

func @main(%arg0: tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) {

// CHECK:  {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:      builtin_code: CUSTOM,
// CHECK-NEXT:      custom_code: "MaxPoolingWithArgmax2D"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    subgraphs: [ {
// CHECK-NEXT:      tensors: [ {
// CHECK-NEXT:        shape: [ 1, 64, 64, 32 ],
// CHECK-NEXT:        buffer: 1,
// CHECK-NEXT:        name: "arg0",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      }, {
// CHECK-NEXT:        shape: [ 1, 32, 32, 32 ],
// CHECK-NEXT:        buffer: 2,
// CHECK-NEXT:        name: "tfl.max_pooling_with_argmax_2d",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 32, 32, 32 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "tfl.max_pooling_with_argmax_2d:1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0 ],
// CHECK-NEXT:    outputs: [ 1, 2 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0 ],
// CHECK-NEXT:      outputs: [ 1, 2 ],
// CHECK-NEXT:      custom_options: [ 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
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

// MLIR-LABEL: func @main(%arg0: tensor<1x64x64x32xf32>)
// MLIR-SAME:    -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
// MLIR:         %value, %indices = "tfl.max_pooling_with_argmax_2d"(%arg0)
// MLIR-SAME:      {filter_h = 4 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 1 : i32}
// MLIR-SAME:      (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
// MLIR-NEXT:    return %value, %indices : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>

  %0, %1 = "tfl.max_pooling_with_argmax_2d"(%arg0) {filter_h = 4 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 1 : i32} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  return %0, %1 : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>
}
