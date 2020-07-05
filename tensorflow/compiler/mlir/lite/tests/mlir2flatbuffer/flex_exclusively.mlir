// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-select-tf-ops=true -emit-builtin-tflite-ops=false -o - | flatbuffer_to_string - | FileCheck %s

func @main(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK:  {
// CHECK-NEXT:    version: 3,
// CHECK-NEXT:    operator_codes: [ {
// CHECK-NEXT:      builtin_code: CUSTOM,
// CHECK-NEXT:      custom_code: "FlexAddV2"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    subgraphs: [ {
// CHECK-NEXT:      tensors: [ {
// CHECK-NEXT:        shape: [ 3, 2 ],
// CHECK-NEXT:        buffer: 1,
// CHECK-NEXT:        name: "arg0",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      }, {
// CHECK-NEXT:        shape: [ 3, 2 ],
// CHECK-NEXT:        buffer: 2,
// CHECK-NEXT:        name: "tf.AddV2",
// CHECK-NEXT:        quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:        }
// CHECK-NEXT:      } ],
// CHECK-NEXT:      inputs: [ 0 ],
// CHECK-NEXT:      outputs: [ 1 ],
// CHECK-NEXT:      operators: [ {
// CHECK-NEXT:        inputs: [ 0, 0 ],
// CHECK-NEXT:        outputs: [ 1 ],
// CHECK-NEXT:        custom_options: [ 5, 65, 100, 100, 86, 50, 0, 22, 18, 5, 65, 100, 100, 86, 50, 26, 0, 26, 0, 42, 7, 10, 1, 84, 18, 2, 48, 1, 50, 0, 0, 2, 31, 25, 20, 20, 4, 40, 1 ]
// CHECK-NEXT:      } ],
// CHECK-NEXT:      name: "main"
// CHECK-NEXT:    } ],
// CHECK-NEXT:    description: "MLIR Converted.",
// CHECK-NEXT:    buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:    }, {
// CHECK-EMPTY:
// CHECK-NEXT:    }, {
// CHECK-EMPTY:
// CHECK-NEXT:    }, {
// CHECK-NEXT:      data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:    } ],
// CHECK-NEXT:    metadata: [ {
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 3
// CHECK-NEXT:    } ]
// CHECK-NEXT:  }

  %0 = "tf.AddV2"(%arg0, %arg0) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
