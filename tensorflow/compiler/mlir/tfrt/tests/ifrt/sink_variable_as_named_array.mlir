// RUN: tf-tfrt-opt -split-input-file -sink-variable-as-named-array %s | FileCheck %s

// -----
// Basic test: all variables tensors are for devices and sinked as named ifrt arrays
//
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:   [[HANDLE2:%.*]] = "tf.VarHandleOp"
// CHECK-NEXT:   [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"([[HANDLE2]])
// CHECK-SAME:       used_by_host = false 
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"([[KEY]], %arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [0 : i32]}>
// CHECK-SAME:    : (tensor<!tf_type.string>, tensor<1x3xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %result = "tf.IfrtCall"(%2, %arg0) <{program_id = 6515870160938153680 : i64, variable_arg_indices = []}> : (tensor<3x1xf32>, tensor<1x3xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// Variable tensor for host can still be used.
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK:  "tf.VarHandleOp"
// CHECK-NOT:  [[VARIABLE:%.*]] = "tf.ReadVariableOp"
// CHECK-NEXT:  [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"
// CHECK-SAME:    used_by_host = true
// CHECK-NEXT:  "tf.MatMul"(%arg0, [[FUTURE]])
// CHECK-NEXT:   [[RES:%.*]] = "tf.IfrtCall"(%arg0, [[KEY]]) <{program_id = 6515870160938153680 : i64, variable_arg_indices = [1 : i32]}>
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %3 = "tf.MatMul"(%arg0, %2) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    %result = "tf.IfrtCall"(%arg0, %2) <{program_id = 6515870160938153680 : i64, variable_arg_indices = []}> : (tensor<1x3xf32>, tensor<3x1xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// Variable tensor is only for host
//
// CHECK-LABEL:  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK:  "tf.VarHandleOp"
// CHECK-NOT:  [[VARIABLE:%.*]] = "tf.ReadVariableOp"
// CHECK-NEXT:  [[KEY:%.*]], [[FUTURE:%.*]] = "tf.IfrtLoadVariable"
// CHECK-SAME:    used_by_host = true
// CHECK-NEXT:  [[RES:%.*]] = "tf.MatMul"(%arg0, [[FUTURE]])
// CHECK-NEXT:    return [[RES]] : tensor<1x1xf32>
//
module {
  func.func @serving_default(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
    %3 = "tf.MatMul"(%arg0, %2) : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    return %3: tensor<1x1xf32>
  }
}

// -----
//  Resources that are created in the same module are not sinked.
//
// CHECK-LABEL:  func.func @serving_default
// CHECK-NOT:  IfrtLoadVariable
// CHECK:      "tf.VarHandleOp"
// CHECK-NEXT: "tf.AssignVariableOp"
// CHECK-NEXT: "tf.ReadVariableOp"
// CHECK-NEXT: "tf.StatefulPartitionedCall"
// CHECK-NEXT:  return 
//
module {
  func.func @serving_default() -> tensor<*xi32> {
    %cst = "tf.Const"() <{value = dense<"some_test.txt"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %0 = "tf.VarHandleOp"() <{container = "", shared_name = "Variable"}> : () -> tensor<!tf_type.resource<tensor<!tf_type.string>>>
    "tf.AssignVariableOp"(%0, %cst) <{validate_shape = false}> : (tensor<!tf_type.resource<tensor<!tf_type.string>>>, tensor<!tf_type.string>) -> ()
    %2 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<!tf_type.string>>>) -> tensor<*x!tf_type.string>
    %4 = "tf.StatefulPartitionedCall"(%2) <{config = "", config_proto = "", executor_type = "", f = @__initializer}> : (tensor<*x!tf_type.string>) -> tensor<*xi32>
    return %4: tensor<*xi32>
  }
  func.func @__initializer(%arg0: tensor<*x!tf_type.string>) -> tensor<i32> {
    %0 = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    return %0 : tensor<i32>
  }
}
