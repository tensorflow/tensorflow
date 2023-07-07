// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s | flatbuffer_to_string - | FileCheck %s

func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "TensorListReserve", custom_option = #tfl<const_bytes : "0x02">} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<*xi32>>>
}

// CHECK:   operator_codes: [ {
// CHECK:       custom_code: "TensorListReserve",
// CHECK:       builtin_code: CUSTOM

// CHECK:         shape: [  ],
// CHECK:         type: VARIANT,
// CHECK:         name: "tfl.custom",

// CHECK:         variant_tensors: [ {
// CHECK:           shape: [  ],
// CHECK:           type: INT32

// CHECK:       operators: [ {
// CHECK:         inputs: [ 0, 1 ],
// CHECK:         outputs: [ 2 ],
// CHECK:         custom_options: [ 2 ]
