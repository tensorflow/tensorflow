// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string - | FileCheck %s

func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %0 = constant unit
  %1:2 = "tfl.fully_connected"(%arg0, %arg1, %0) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  return %1 : tensor<40x40xf32>
}

// CHECK: operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, -1 ],
// CHECK-NEXT:       outputs: [ 2, 3 ],
// CHECK-NEXT:       builtin_options_type: FullyConnectedOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
