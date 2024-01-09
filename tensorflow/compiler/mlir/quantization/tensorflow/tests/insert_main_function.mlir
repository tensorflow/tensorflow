// RUN: tf-quant-opt %s -quant-insert-main-function -mlir-disable-threading \
// RUN:     -allow-unregistered-dialect -split-input-file | FileCheck %s

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
// CHECK:   %[[CONST_0:.*]] = "tf.Const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK:   %[[MUL_1:.*]] = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[MUL_2:.*]] = "tf.Mul"(%[[MUL_1]], %[[CONST_0]]) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK:   return %[[MUL_2]] : tensor<1xf32>
// CHECK: }

// CHECK: func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["mul1_y:0"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["mul1_x:0"]}, %arg2: tensor<1xf32> {tf_saved_model.index_path = ["mul2_y:0"]}, %arg3: tensor<1xf32> {tf_saved_model.index_path = ["mul2_x:0"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall:0"]}, tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall_1:0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0,mul2_y:0,mul2_x:0", outputs = "PartitionedCall:0,PartitionedCall_1:0"}, tf_saved_model.exported_names = ["main"]} {
// CHECK-NOT: f = @NoOp
// CHECK:   %[[PARTITIONEDCALL_0:.*]] = "tf.PartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "", executor_type = "", f = @mul1}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:   %[[PARTITIONEDCALL_1:.*]] = "tf.PartitionedCall"(%arg2, %arg3) <{config = "", config_proto = "", executor_type = "", f = @mul2}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK-DAG:   %[[IDENTITY_0:.*]] = "tf.Identity"(%[[PARTITIONEDCALL_0]])
// CHECK-DAG:   %[[IDENTITY_1:.*]] = "tf.Identity"(%[[PARTITIONEDCALL_1]])
// CHECK:   return %[[IDENTITY_0]], %[[IDENTITY_1]] : tensor<1xf32>, tensor<1xf32>
// CHECK: }
}

// -----

// Test a case where there is an exported function not labeled tf.entry_function.
// CHECK-LABEL: module attributes {tf.versions = {producer = 1132 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
module attributes {tf.versions = {producer = 1132 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  "tf_saved_model.asset"() {filename = "assets/mydata.txt", sym_name = "__tf_saved_model_asset0_mydata.txt"} : () -> ()
// Session initializer ops and asset ops untouched.
// CHECK: "tf_saved_model.session_initializer"() <{initializers = [@NoOp]}> : () -> ()
// CHECK: "tf_saved_model.asset"() <{filename = "assets/mydata.txt", sym_name = "__tf_saved_model_asset0_mydata.txt"}> : () -> ()

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
// CHECK: %[[CALL0:.*]] = "tf.PartitionedCall"(%[[ARG0]], %[[ARG1]]) <{
// CHECK-NOT: f = @NoOp
// CHECK-SAME: f = @add
// CHECK-SAME: }>
// CHECK-SAME: : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[CALL0]])
// CHECK: return %[[IDENTITY]] : tensor<1xf32>
}

// -----

// Test a case where an entry function return multiple values
module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    func.return
  }

  func.func @topk(%arg0: tensor<16xf32> {tf_saved_model.index_path = ["input"]}, %arg1: tensor<i32> {tf_saved_model.index_path = ["k"]}) -> (tensor<?xf32> {tf_saved_model.index_path = ["values"]}, tensor<?xi32> {tf_saved_model.index_path = ["indices"]}) attributes {tf.entry_function = {inputs = "input:0,k:0", outputs = "TopK:0,TopK:1"}, tf_saved_model.exported_names = ["topk"]} {
    %0:2 = "tf.TopKV2"(%arg0, %arg1): (tensor<16xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
    func.return %0#0, %0#1: tensor<?xf32>, tensor<?xi32>
  }

// CHECK: func.func private @topk(%arg0: tensor<16xf32>, %arg1: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
// CHECK-SAME: attributes {tf.entry_function = {inputs = "input:0,k:0", outputs = "TopK:0,TopK:1"}}

// CHECK: func.func @main(%arg0: tensor<16xf32> {tf_saved_model.index_path = ["input:0"]}, %arg1: tensor<i32> {tf_saved_model.index_path = ["k:0"]})
// CHECK-SAME: -> (tensor<?xf32> {tf_saved_model.index_path = ["TopK:0"]}, tensor<?xi32> {tf_saved_model.index_path = ["TopK:1"]})
// CHECK-SAME: attributes {tf.entry_function = {inputs = "input:0,k:0", outputs = "TopK:0,TopK:1"}, tf_saved_model.exported_names = ["main"]}
// CHECK: %[[CALL0:.*]]:2 = "tf.PartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "", executor_type = "", f = @topk}>
// Expects an IdentityN op to be created.
// CHECK: %[[IDENTITY:.*]]:2 = "tf.IdentityN"(%[[CALL0]]#0, %[[CALL0]]#1) : (tensor<?xf32>, tensor<?xi32>) -> (tensor<?xf32>, tensor<?xi32>)
// CHECK: return %[[IDENTITY]]#0, %[[IDENTITY]]#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

// Test that the signature prefix is added when there are duplicated input names.
module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    func.return
  }

  func.func @mul1(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "y:0,x:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["mul1"]} {
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }

  func.func @mul2(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "y:0,x:0", outputs = "PartitionedCall_1:0"}, tf_saved_model.exported_names = ["mul2"]} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
    func.return %1 : tensor<1xf32>
  }

// CHECK: func @main
// CHECK: (%arg0: tensor<1xf32> {tf_saved_model.index_path = ["mul1_y:0"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["mul1_x:0"]}
// CHECK: %arg2: tensor<1xf32> {tf_saved_model.index_path = ["mul2_y:0"]}, %arg3: tensor<1xf32> {tf_saved_model.index_path = ["mul2_x:0"]})
// CHECK: -> (tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall:0"]}, tensor<1xf32> {tf_saved_model.index_path = ["PartitionedCall_1:0"]})
// CHECK: attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0,mul2_y:0,mul2_x:0", outputs = "PartitionedCall:0,PartitionedCall_1:0"}, tf_saved_model.exported_names = ["main"]}
}

// -----

// Test that the signature prefix is added when there are duplicated output names.
module attributes {tf.versions = {producer = 930 : i32}, tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32}  {
  "tf_saved_model.session_initializer"() {initializers = [@NoOp]} : () -> ()
  func.func @NoOp() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_NoOp"]} {
    func.return
  }

  func.func @mul1(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0", outputs = "output:0"}, tf_saved_model.exported_names = ["mul1"]} {
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }

  func.func @mul2(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["y"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "mul2_y:0,mul2_x:0", outputs = "output:0"}, tf_saved_model.exported_names = ["mul2"]} {
    %cst = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Mul"(%arg1, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
    func.return %1 : tensor<1xf32>
  }
// CHECK: func @main
// CHECK: (%arg0: tensor<1xf32> {tf_saved_model.index_path = ["mul1_y:0"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["mul1_x:0"]}
// CHECK: %arg2: tensor<1xf32> {tf_saved_model.index_path = ["mul2_y:0"]}, %arg3: tensor<1xf32> {tf_saved_model.index_path = ["mul2_x:0"]})
// CHECK: -> (tensor<1xf32> {tf_saved_model.index_path = ["mul1_output:0"]}, tensor<1xf32> {tf_saved_model.index_path = ["mul2_output:0"]})
// CHECK: attributes {tf.entry_function = {inputs = "mul1_y:0,mul1_x:0,mul2_y:0,mul2_x:0", outputs = "mul1_output:0,mul2_output:0"}, tf_saved_model.exported_names = ["main"]}
}

// -----

// Tests when a function called @main already exists, it is renamed to
// `main_{i}` to avoid conflict.
module attributes {tf_saved_model.semantics}  {
  func.func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["y"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "x:0,y:0", outputs = "output:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }

// CHECK: func.func private @main_0
// CHECK: func.func @main
}

// -----

// Tests when a function called @main already exists and @main_{i} also already
// exists, it increments the suffix number until there's no conflict.
module attributes {tf_saved_model.semantics}  {
  func.func @main_0(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["z"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "z:0", outputs = "output:0"}, tf_saved_model.exported_names = ["main_0"]} {
    %0 = "tf.Identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }

  func.func @main(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<1xf32> {tf_saved_model.index_path = ["y"]}) -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "x:0,y:0", outputs = "output:0"}, tf_saved_model.exported_names = ["main"]} {
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }
// `@main_0` remains touched.
// CHECK: func.func private @main_0
// CHECK-SAME: z:0

// `@main` should be renamed to `@main_1` instead of `@main_0` to avoid
// conflict.
// CHECK: func.func private @main_1
// CHECK-SAME: x:0

// This is the newly created main function.
// CHECK: func.func @main
}
