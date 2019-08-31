// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_to_string -
// | FileCheck %s

func @main(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %cst = constant unit
  %0 = "tfl.pseudo_input"(%arg0) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %1 = "tfl.pseudo_input"(%arg1) : (tensor<40x37xf32>) -> tensor<40x37xf32> loc("Input")
  %2:2 = "tfl.fully_connected"(%0, %1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> (tensor<40x40xf32>, tensor<40x40xf32>)
  return %2 : tensor<40x40xf32>
}

// CHECK-NEXT: operators: [ {
// CHECK-NEXT:       inputs: [ 0, 1, -1 ],
// CHECK-NEXT:       outputs: [ 2, 3 ],
// CHECK-NEXT:       builtin_options_type: FullyConnectedOptions,
// CHECK-NEXT:       builtin_options: {
// CHECK-EMPTY:
// CHECK-NEXT:       }
// CHECK-NEXT:     } ],
