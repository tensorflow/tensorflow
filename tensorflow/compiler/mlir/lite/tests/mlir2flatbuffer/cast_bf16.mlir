// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-custom-ops -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(tensor<4x5xbf16>) -> tensor<4x5xbf16> {
^bb0(%arg0: tensor<4x5xbf16>):

// CHECK:  {
// CHECK-NEXT:      version: 3,
// CHECK-NEXT:      operator_codes: [ {
// CHECK-NEXT:        deprecated_builtin_code: 53,
// CHECK-NEXT:        version: 7,
// CHECK-NEXT:        builtin_code: CAST
// CHECK-NEXT:      } ],
// CHECK-NEXT:      subgraphs: [ {
// CHECK-NEXT:        tensors: [ {
// CHECK-NEXT:          shape: [ 4, 5 ],
// CHECK-NEXT:          type: BFLOAT16,
// CHECK-NEXT:          buffer: 1,
// CHECK-NEXT:          name: "arg0",
// CHECK-NEXT:          quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:          },
// CHECK-NEXT:          has_rank: true
// CHECK-NEXT:        }, {
// CHECK-NEXT:          shape: [ 4, 5 ],
// CHECK-NEXT:          buffer: 2,
// CHECK-NEXT:          name: "cast1",
// CHECK-NEXT:          quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:          },
// CHECK-NEXT:          has_rank: true
// CHECK-NEXT:        }, {
// CHECK-NEXT:          shape: [ 4, 5 ],
// CHECK-NEXT:          type: BFLOAT16,
// CHECK-NEXT:          buffer: 3,
// CHECK-NEXT:          name: "cast2",
// CHECK-NEXT:          quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:          },
// CHECK-NEXT:          has_rank: true
// CHECK-NEXT:        } ],
// CHECK-NEXT:        inputs: [ 0 ],
// CHECK-NEXT:        outputs: [ 2 ],
// CHECK-NEXT:        operators: [ {
// CHECK-NEXT:          inputs: [ 0 ],
// CHECK-NEXT:          outputs: [ 1 ]
// CHECK-NEXT:        }, {
// CHECK-NEXT:          inputs: [ 1 ],
// CHECK-NEXT:          outputs: [ 2 ]
// CHECK-NEXT:        } ],
// CHECK-NEXT:        name: "main"
// CHECK-NEXT:      } ],
// CHECK-NEXT:      description: "MLIR Converted.",
// CHECK-NEXT:      buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:      }, {
// CHECK-EMPTY:
// CHECK-NEXT:      }, {
// CHECK-EMPTY:
// CHECK-NEXT:      }, {
// CHECK-EMPTY:
// CHECK-NEXT:      }, {
// CHECK-NEXT:        data: [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
// CHECK-NEXT:      } ],
// CHECK-NEXT:      metadata: [ {
// CHECK-NEXT:        name: "min_runtime_version",
// CHECK-NEXT:        buffer: 4
// CHECK-NEXT:      } ],
// CHECK-NEXT:      signature_defs: [  ]
// CHECK-NEXT:    }

  %0 = "tfl.cast" (%arg0) : (tensor<4x5xbf16>) -> tensor<4x5xf32> loc("cast1")
  %1 = "tfl.cast" (%0) : (tensor<4x5xf32>) -> tensor<4x5xbf16> loc("cast2")
  func.return %1 : tensor<4x5xbf16>
}
