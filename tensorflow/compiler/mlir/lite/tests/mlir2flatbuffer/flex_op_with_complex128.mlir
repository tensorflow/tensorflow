// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -emit-select-tf-ops -o - | flatbuffer_to_string - | FileCheck %s

func.func @main(tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
^bb0(%arg0: tensor<4xcomplex<f64>>, %arg1: tensor<4xcomplex<f64>>):
// CHECK:  {
// CHECK-NEXT:  version: 3,
// CHECK-NEXT:  operator_codes: [ {
// CHECK-NEXT:    deprecated_builtin_code: 32,
// CHECK-NEXT:    custom_code: "FlexAdd",
// CHECK-NEXT:    builtin_code: CUSTOM
// CHECK-NEXT:  } ],
// CHECK-NEXT:  subgraphs: [ {
// CHECK-NEXT:    tensors: [ {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: COMPLEX128,
// CHECK-NEXT:      buffer: 1,
// CHECK-NEXT:      name: "arg0",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: COMPLEX128,
// CHECK-NEXT:      buffer: 2,
// CHECK-NEXT:      name: "arg1",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    }, {
// CHECK-NEXT:      shape: [ 4 ],
// CHECK-NEXT:      type: COMPLEX128,
// CHECK-NEXT:      buffer: 3,
// CHECK-NEXT:      name: "add",
// CHECK-NEXT:      quantization: {
// CHECK-EMPTY:
// CHECK-NEXT:      },
// CHECK-NEXT:      has_rank: true
// CHECK-NEXT:    } ],
// CHECK-NEXT:    inputs: [ 0, 1 ],
// CHECK-NEXT:    outputs: [ 2 ],
// CHECK-NEXT:    operators: [ {
// CHECK-NEXT:      inputs: [ 0, 1 ],
// CHECK-NEXT:      outputs: [ 2 ],
// CHECK-NEXT:      custom_options: [ 3, 65, 100, 100, 0, 20, 18, 3, 65, 100, 100, 26, 0, 26, 0, 42, 7, 10, 1, 84, 18, 2, 48, 18, 50, 0, 0, 2, 27, 23, 20, 20, 4, 40, 1 ]
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
// CHECK-NEXT:    name: "min_runtime_version",
// CHECK-NEXT:    buffer: 4
// CHECK-NEXT:  } ]
// CHECK-NEXT:  signature_defs: [ ]
// CHECK-NEXT:}

  %0 = "tf.Add"(%arg0, %arg1)  : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> loc("add")
  func.return %0 : tensor<4xcomplex<f64>>
}
