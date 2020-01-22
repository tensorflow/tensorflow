// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -o - | flatbuffer_to_string - | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir -o - | FileCheck --check-prefix=MLIR %s


func @main(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<1x32x42x128xf32>, %arg2: tensor<4xi32>) -> tensor<1x64x84x32xf32> {

// CHECK:  {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:      builtin_code: CUSTOM,
// CHECK-NEXT:      custom_code: "Convolution2DTransposeBias"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    subgraphs: [ {
// CHECK-NEXT:      tensors: [ {
// CHECK-NEXT:        shape: [ 32, 4, 4, 128 ],
// CHECK-NEXT:        buffer: 1,
// CHECK-NEXT:        name: "arg0",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      }, {
// CHECK-NEXT:        shape: [ 1, 32, 42, 128 ],
// CHECK-NEXT:        buffer: 2,
// CHECK-NEXT:        name: "arg1",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "arg2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 64, 84, 32 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "tfl.convolution_2d_transpose_bias",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1, 2 ],
// CHECK-NEXT:    outputs: [ 3 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1, 2 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      custom_options: [ 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0 ]
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
// CHECK-EMPTY:
// CHECK-NEXT:  } ]
// CHECK-NEXT:}

// MLIR-LABEL: func @main(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<1x32x42x128xf32>, %arg2: tensor<4xi32>)
// MLIR-SAME:    -> tensor<1x64x84x32xf32>
// MLIR:         %0 = "tfl.convolution_2d_transpose_bias"(%arg0, %arg1, %arg2)
// MLIR-SAME:      {padding = "SAME", stride_h = 1 : i32, stride_w = 2 : i32}
// MLIR-SAME:      (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
// MLIR-NEXT:    return %0 : tensor<1x64x84x32xf32>

  %0 = "tfl.convolution_2d_transpose_bias"(%arg0, %arg1, %arg2) {padding = "SAME", stride_h = 1 : i32, stride_w = 2 : i32} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
  return %0 : tensor<1x64x84x32xf32>
}
