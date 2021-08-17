// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// CHECK: func @add(
// CHECK-SAME: {tf_saved_model.index_path = ["input1"]}
// CHECK-SAME: {tf_saved_model.index_path = ["input2"]}
// CHECK-SAME: {tf_saved_model.index_path = ["result"]}
// CHECK-SAME: tf.entry_function = {inputs = "input1:0,input2:0", outputs = "result:0"}
// CHECK-SAME: tf_saved_model.exported_names = ["add"]

// CHECK: func @sub(
// CHECK-SAME: {tf_saved_model.index_path = ["input2"]}
// CHECK-SAME: {tf_saved_model.index_path = ["input1"]}
// CHECK-SAME: {tf_saved_model.index_path = ["result"]}
// CHECK-SAME: tf.entry_function = {inputs = "input2:0,input1:0", outputs = "result:0"}
// CHECK-SAME: tf_saved_model.exported_names = ["sub"]

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 554 : i32}, tf_saved_model.semantics} {
  func @add(%arg0: tensor<?xf32> {tf_saved_model.index_path = ["input1"]}, %arg1: tensor<?xf32> {tf_saved_model.index_path = ["input2"]}) -> (tensor<?xf32> {tf_saved_model.index_path = ["result"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "input1:0,input2:0", outputs = "result:0"}, tf_saved_model.exported_names = ["add"]} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?xf32>
    %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<?xf32>
    return %1 : tensor<?xf32>
  }

  func @sub(%arg0: tensor<?xf32> {tf_saved_model.index_path = ["input2"]}, %arg1: tensor<?xf32> {tf_saved_model.index_path = ["input1"]}) -> (tensor<?xf32> {tf_saved_model.index_path = ["result"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "input2:0,input1:0", outputs = "result:0"}, tf_saved_model.exported_names = ["sub"]} {
    %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}
