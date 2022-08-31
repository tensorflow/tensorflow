// RUN: tf-quant-opt %s -quant-add-main-function -allow-unregistered-dialect -mlir-disable-threading -split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    func.return
  }
// CHECK: func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]}

  func.func @mul1(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["mul1"]} {
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
// CHECK: func private @mul1(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0", outputs = "PartitionedCall:0"}}
// CHECK:   %[[MUL_0:.*]] = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   return %[[MUL_0]] : tensor<1xf32>
// CHECK: }

  func.func @mul2(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul2_y:0,mul2_x:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["mul2"]} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
    func.return %1 : tensor<1xf32>
  }
// CHECK: func private @mul2(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {tf.entry_function = {inputs = "mul2_y:0,mul2_x:0", outputs = "PartitionedCall_1:0"}} {
// CHECK:   %[[CONST_0:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK:   %[[MUL_1:.*]] = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[MUL_2:.*]] = "tf.Mul"(%[[MUL_1]], %[[CONST_0]]) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK:   return %[[MUL_2]] : tensor<1xf32>
// CHECK: }

// CHECK: func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["mul1_y:0"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["mul1_x:0"]}, %arg2: tensor<1xf32> {tf_saved_model.index_path = ["mul2_y:0"]}, %arg3: tensor<1xf32> {tf_saved_model.index_path = ["mul2_x:0"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall:0"]}, tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall_1:0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0,mul2_y:0,mul2_x:0", outputs = "PartitionedCall:0,PartitionedCall_1:0"}, tf_saved_model.exported_names = ["main"]} {
// CHECK-NOT: f = @NoOp
// CHECK:   %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @mul1} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg2, %arg3) {config = "", config_proto = "", executor_type = "", f = @mul2} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   return %[[PARTITIONEDCALL_0]], %[[PARTITIONEDCALL_1]] : tensor<1xf32>, tensor<1xf32>
// CHECK: }
}

// -----

// Test a case where there is an exported function not labeled tf.entry_function.
// CHECK-LABEL: module attributes {tf.versions = {producer = 1132 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
module attributes {tf.versions = {producer = 1132 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/mydata.txt", sym_name = "__tf_saved_model_asset0_mydata.txt"} : () -> ()
// Session initializer ops and asset ops untouched.
// CHECK: "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
// CHECK: "tf_saved_model.asset"() {filename = "assets/mydata.txt", sym_name = "__tf_saved_model_asset0_mydata.txt"} : () -> ()

  func.func @NoOp(%arg0: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_mydata.txt}) attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    "tf.InitializeTableFromTextFileV2"(%0, %arg0) {delimiter = "\09", device = "", key_index = -2 : i64, offset = 0 : i64, value_index = -1 : i64, vocab_size = 437 : i64} : (tensor<!tf_type.resource>, tensor<!tf_type.string>) -> ()
    func.return
  }
// Initializer function untouched.
// CHECK: func.func @NoOp(%[[ARG0:.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @__tf_saved_model_asset0_mydata.txt})
// CHECK-SAME: {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]}
// CHECK: %[[HASH_TABLE0:.*]] = "tf.HashTableV2"()
// CHECK: "tf.InitializeTableFromTextFileV2"(%[[HASH_TABLE0]], %[[ARG0]])
// CHECK: return

  func.func @add(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["y"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["out_0"]}) attributes {tf.entry_function = {inputs = "add_x:0,add_y:0", outputs = "add:0"}, tf_saved_model.exported_names = ["add"]} {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
// The previously exported function should now be private.
// CHECK: func.func private @add
// CHECK-NOT: tf_saved_model.exported_names
// Other attributes should be left untouched.
// CHECK-SAME: attributes {tf.entry_function = {inputs = "add_x:0,add_y:0", outputs = "add:0"}}

// Test the newly created "main" function.
// CHECK: func.func @main(%[[ARG0:.*]]: tensor<1xf32> {tf_saved_model.index_path = ["add_x:0"]}, %[[ARG1:.*]]: tensor<1xf32> {tf_saved_model.index_path = ["add_y:0"]})
// CHECK-SAME: -> (tensor<1xf32> {tf_saved_model.index_path = ["add:0"]})
// Check attributes of the main function.
// CHECK-SAME: tf.entry_function = {inputs = "add_x:0,add_y:0", outputs = "add:0"}
// CHECK-SAME: tf_saved_model.exported_names = ["main"]

// Check that the function call to @add exists and not to @NoOp.
// CHECK: %[[CALL0:.*]] = "tf.PartitionedCall"(%[[ARG0]], %[[ARG1]]) {
// CHECK-NOT: f = @NoOp
// CHECK-SAME: f = @add
// CHECK-SAME: }
// CHECK-SAME: : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[CALL0]] : tensor<1xf32>
}
