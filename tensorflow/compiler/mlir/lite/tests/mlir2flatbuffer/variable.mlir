// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func.func @main() -> tensor<3x2xi32> {
  %0 = "tfl.pseudo_const" () {value = dense<0> : tensor<3x2xi32>, tfl.is_variable} : () -> tensor<3x2xi32> loc("variable")
  func.return %0 : tensor<3x2xi32>
}

// CHECK:      {
// CHECK-NEXT:     version: 3,
// CHECK-NEXT:     operator_codes: [ ],
// CHECK-NEXT:     subgraphs: [ {
// CHECK-NEXT:       tensors: [ {
// CHECK-NEXT:         shape: [ 3, 2 ],
// CHECK-NEXT:         type: INT32,
// CHECK-NEXT:         name: "variable",
// CHECK-NEXT:         quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:         },
// CHECK-NEXT:         is_variable: true
// CHECK-NEXT:         has_rank: true
// CHECK-NEXT:       } ],
// CHECK-NEXT:       inputs: [ ],
// CHECK-NEXT:       outputs: [ 0 ],
// CHECK-NEXT:       operators: [ ],
// CHECK-NEXT:       name: "main"
// CHECK-NEXT:     } ],
// CHECK-NEXT:     description: "MLIR Converted.",
// CHECK-NEXT:     buffers: [ {
// CHECK-EMPTY:
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ {{.*}} ]
// CHECK-NEXT:     }, {
// CHECK-NEXT:      data: [ {{.*}} ]
// CHECK-NEXT:     } ],
// CHECK-NEXT:     metadata: [ {
// CHECK-NEXT:     name: "min_runtime_version",
// CHECK-NEXT:     buffer: 2
// CHECK-NEXT:     } ]
// CHECK-NEXT:     signature_defs: [ ]
// CHECK-NEXT:   }