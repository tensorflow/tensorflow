// RUN: tf-opt %s -split-input-file -tfl-analyze-variables-pass --cse | FileCheck %s

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func @f() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<*xi32>>>
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32>
    return %2 : tensor<*xi32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = true}
module {
  func @main() -> tensor<*xi32> {
    %0 = "tf.PartitionedCall"() {f = @f, config = "", config_proto = "", executor_type = ""}
      : () -> tensor<*xi32>
    return %0 : tensor<*xi32>
  }
  func @f() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<*xi32>>>
    %1 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32>
    return %1 : tensor<*xi32>
  }
}


// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = false}
module {
  func @main() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<*xi32>>>
    %1 = "tf.PartitionedCall"(%0) {f = @f, config = "", config_proto = "", executor_type = ""}
      : (tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32>
    return %1 : tensor<*xi32>
  }
  func @f(%arg0 : tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32> {
    %0 = "tf.ReadVariableOp"(%arg0) {dtype = i32} : (tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32>
    return %0 : tensor<*xi32>
  }
}

// -----

// CHECK: module attributes {tfl._legalize_tfl_variables = false}
module {
  func @main() -> tensor<*xi32> {
    %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<*xi32>>>
    %cst = constant dense<2> : tensor<4xi32>
    "tf.AssignAddVariableOp"(%0, %cst) {} : (tensor<*x!tf.resource<tensor<*xi32>>>, tensor<4xi32>) -> ()
    %1 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource<tensor<*xi32>>>) -> tensor<*xi32>
    return %1 : tensor<*xi32>
  }
}
