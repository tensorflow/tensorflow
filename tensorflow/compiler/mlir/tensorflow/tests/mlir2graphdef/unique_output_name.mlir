// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

module attributes {tf.versions = {producer = 946 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  func @main(%arg0: tensor<2x4x2x2xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<2x2x1x2xi32> {tf_saved_model.index_path = ["output_1"]}, tensor<2x2x1x2xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_x:0", outputs = "PartitionedCall:1,PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0:2 = tf_executor.graph {
      %outputs:2, %control = tf_executor.island wraps "tf.MaxPoolWithArgmax"(%arg0) {T = f32, Targmax = i32, include_batch_in_index = false, ksize = [1, 2, 2, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<2x4x2x2xf32>) -> (tensor<2x2x1x2xf32>, tensor<2x2x1x2xi32>)
      tf_executor.fetch %outputs#1, %outputs#0 : tensor<2x2x1x2xi32>, tensor<2x2x1x2xf32>
    }
    return %0#0, %0#1 : tensor<2x2x1x2xi32>, tensor<2x2x1x2xf32>
  }
}

// CHECK:        name: "serving_default_x"
// CHECK-NEXT:   op: "_Arg"

// CHECK:        name: "tf.MaxPoolWithArgmax"
// CHECK-NEXT:   op: "MaxPoolWithArgmax"
// CHECK-NEXT:   input: "serving_default_x"


// CHECK:        name: "PartitionedCall"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "tf.MaxPoolWithArgmax:1"

// CHECK:        name: "PartitionedCall1"
// CHECK-NOT:    name: "PartitionedCall"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "tf.MaxPoolWithArgmax"
