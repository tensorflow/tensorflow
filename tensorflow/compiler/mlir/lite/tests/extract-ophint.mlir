// RUN: tf-opt -tfl-extract-ophint %s | FileCheck %s

// CHECK-LABEL: extractSimpleOphint
func @extractSimpleOphint() {
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @d4b1eb00b81211e99426dc4a3e957995(%0) : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
// CHECK:  %[[OUTPUT:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "d4b1eb00b81211e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation-d4b1eb00b81211e99426dc4a3e957995-0-None-None"} : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation", _tflite_function_uuid = "d4b1eb00b81211e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation-d4b1eb00b81211e99426dc4a3e957995-0-None-None"} : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
  %3 = "tf.Mul"(%2, %1) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x16x1xf32>, tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "d4b1eb00b81211e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation-d4b1eb00b81211e99426dc4a3e957995-0-None-None"} : (tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
  return
}

// CHECK-LABEL: extractPackedInputOphint
func @extractPackedInputOphint() {
// CHECK:  %[[PACK:[0-9]*]] = "tfl.pack"(%0, %1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<2x1x16x1xf32>
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @47393154b9af11e99426dc4a3e957995(%[[PACK]]) : (tensor<2x1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_stack", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "47393154b9af11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack-47393154b9af11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_stack", _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "47393154b9af11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_stack-47393154b9af11e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder_1", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_stack", _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "47393154b9af11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_stack-47393154b9af11e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_1"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %6 = "tf.Mul"(%2, %5) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %7 = "tf.Identity"(%6) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_stack", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "47393154b9af11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack-47393154b9af11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK-LABEL: extractFirstInputOphint
func @extractFirstInputOphint() {
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @b703f0f4b9ec11e99426dc4a3e957995(%0) : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_first", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "b703f0f4b9ec11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_first-b703f0f4b9ec11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "first", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_first", _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "b703f0f4b9ec11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_first-b703f0f4b9ec11e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder_1", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "first", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_first", _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "b703f0f4b9ec11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_first-b703f0f4b9ec11e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_1"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %6 = "tf.Mul"(%2, %5) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %7 = "tf.Identity"(%6) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_first", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "b703f0f4b9ec11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_first-b703f0f4b9ec11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK-LABEL: extractLastInputOphint
func @extractLastInputOphint() {
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @e31fcf90b9ed11e99426dc4a3e957995(%1) : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_last", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "e31fcf90b9ed11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_last-e31fcf90b9ed11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "last", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_last", _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "e31fcf90b9ed11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_last-e31fcf90b9ed11e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder_1", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "last", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_last", _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "e31fcf90b9ed11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_last-e31fcf90b9ed11e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_1"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %6 = "tf.Mul"(%2, %5) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %7 = "tf.Identity"(%6) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_last", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "e31fcf90b9ed11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_last-e31fcf90b9ed11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK-LABEL: extractPackOneInputOphint
func @extractPackOneInputOphint() {
// CHECK:  %[[RESHAPE:[0-9]*]] = "tfl.reshape"(%0) : (tensor<1x16x1xf32>) -> tensor<1x1x16x1xf32>
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @33fab028b9ef11e99426dc4a3e957995(%[[RESHAPE]]) : (tensor<1x1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_pack_input_one", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "33fab028b9ef11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_pack_input_one-33fab028b9ef11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_pack_input_one", _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "33fab028b9ef11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_pack_input_one-33fab028b9ef11e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Identity"(%2) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_pack_input_one", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "33fab028b9ef11e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_pack_input_one-33fab028b9ef11e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK-LABEL: extractStackInputOutputOphint
func @extractStackInputOutputOphint() {
// CHECK:  %[[PACK:[0-9]*]] = "tfl.pack"(%0, %1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<2x1x16x1xf32>
// CHECK:  %[[OP_HINT_CALL:[0-9]*]] = call @b92ed354b9f011e99426dc4a3e957995(%[[PACK]]) : (tensor<2x1x16x1xf32>) -> tensor<2x1x16x1xf32>
// CHECK:  %[[UNPACK:[0-9]*]]:2 = "tfl.unpack"(%[[OP_HINT_CALL]]) {axis = 0 : i32, num = 2 : i32} : (tensor<2x1x16x1xf32>) -> (tensor<1x16x1xf32>, tensor<1x16x1xf32>)
// CHECK:  %[[OUTPUT1:[0-9]*]] = "tf.Identity"(%[[UNPACK]]#0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT2:[0-9]*]] = "tf.Identity"(%[[UNPACK]]#1) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder_1", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_1"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %6 = "tf.Mul"(%2, %5) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %7 = "tf.Identity"(%6) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_sort_index = 0 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-0-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %8 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_2"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %9 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_3"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %10 = "tf.Add"(%8, %9) {T = "tfdtype$DT_FLOAT", name = "add"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %11 = "tf.Identity"(%10) {T = "tfdtype$DT_FLOAT", _tflite_function_aggregate = "stack", _tflite_function_name = "cool_activation_stack_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_sort_index = 1 : i64, _tflite_function_uuid = "b92ed354b9f011e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_stack_input_output-b92ed354b9f011e99426dc4a3e957995-0-1-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK-LABEL: extractMultipleInputsOutputsOphint
func @extractMultipleInputsOutputsOphint() {
// CHECK:   %[[OP_HINT_CALL:[0-9]*]]:2 = call @a6ca45beb9f411e99426dc4a3e957995(%0, %1) : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> (tensor<1x16x1xf32>, tensor<1x16x1xf32>)
// CHECK:  %[[OUTPUT1:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]#0) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:  %[[OUTPUT2:[0-9]*]] = "tf.Identity"(%[[OP_HINT_CALL]]#1) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_output_index = 1 : i64, _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-1-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>

  %0 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %1 = "tf.Identity"(%0) {T = "tfdtype$DT_FLOAT", _tflite_function_input_index = 0 : i64, _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %2 = "tf.Sigmoid"(%1) {T = "tfdtype$DT_FLOAT", name = "Sigmoid"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %3 = "tf.Placeholder"() {dtype = "tfdtype$DT_FLOAT", name = "Placeholder_1", shape = "tfshape$dim { size: 1 } dim { size: 16 } dim { size: 1 }"} : () -> tensor<1x16x1xf32>
  %4 = "tf.Identity"(%3) {T = "tfdtype$DT_FLOAT", _tflite_function_input_index = 1 : i64, _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "InputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-1-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_1"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %6 = "tf.Mul"(%2, %5) {T = "tfdtype$DT_FLOAT", name = "mul"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %7 = "tf.Identity"(%6) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_output_index = 0 : i64, _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-0-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %8 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_2"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %9 = "tf.Sigmoid"(%4) {T = "tfdtype$DT_FLOAT", name = "Sigmoid_3"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %10 = "tf.Add"(%8, %9) {T = "tfdtype$DT_FLOAT", name = "add"} : (tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %11 = "tf.Identity"(%10) {T = "tfdtype$DT_FLOAT", _tflite_function_name = "cool_activation_multiple_input_output", _tflite_function_output_index = 1 : i64, _tflite_function_uuid = "a6ca45beb9f411e99426dc4a3e957995", _tflite_ophint_level = 1 : i64, name = "OutputHint-cool_activation_multiple_input_output-a6ca45beb9f411e99426dc4a3e957995-1-None-None"} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  return
}

// CHECK:  func @d4b1eb00b81211e99426dc4a3e957995(tensor<1x16x16x1xf32>) -> tensor<1x16x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation"}
// CHECK:  func @47393154b9af11e99426dc4a3e957995(tensor<2x1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation_stack"}
// CHECK:  func @b703f0f4b9ec11e99426dc4a3e957995(tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation_first"}
// CHECK:  func @e31fcf90b9ed11e99426dc4a3e957995(tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation_last"}
// CHECK:  func @33fab028b9ef11e99426dc4a3e957995(tensor<1x1x16x1xf32>) -> tensor<1x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation_pack_input_one"}
// CHECK:  func @b92ed354b9f011e99426dc4a3e957995(tensor<2x1x16x1xf32>) -> tensor<2x1x16x1xf32>
// CHECK:    attributes  {_tflite_function_name = "cool_activation_stack_input_output"}
// CHECK:  func @a6ca45beb9f411e99426dc4a3e957995(tensor<1x16x1xf32>, tensor<1x16x1xf32>) -> (tensor<1x16x1xf32>, tensor<1x16x1xf32>)
// CHECK:    attributes  {_tflite_function_name = "cool_activation_multiple_input_output"}