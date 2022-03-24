// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
// CHECK: {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 67,
// CHECK-NEXT:    version: 1,
// CHECK-NEXT:    builtin_code: TRANSPOSE_CONV
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: INT32,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 32, 4, 4, 128 ],
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 32, 42, 128 ],
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "arg2",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 1, 64, 84, 32 ],
// CHECK-NEXT:      buffer: 4,
// CHECK-NEXT:      name: "tfl.transpose_conv",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      }
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1, 2 ],
// CHECK-NEXT:    outputs: [ 3 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1, 2 ],
// CHECK-NEXT:      outputs: [ 3 ],
// CHECK-NEXT:      builtin_options_type: TransposeConvOptions,
// CHECK-NEXT:      builtin_options: {
// CHECK-NEXT:        stride_w: 2,
// CHECK-NEXT:        stride_h: 2
// CHECK-NEXT:      }
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
// CHECK-NEXT:  }, {
// CHECK-NEXT:    data: [ 49, 46, 57, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:  } ],
// CHECK-NEXT:  metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 5
// CHECK-NEXT:  } ]
// CHECK-NEXT:  signature_defs: [ ]
// CHECK-NEXT:}

  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  func.return %0 : tensor<1x64x84x32xf32>
}
