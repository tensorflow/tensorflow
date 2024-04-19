// RUN: tf-opt %s --tf-hoist-loop-invariant | FileCheck %s --dump-input=fail

// CHECK-LABEL: hoist_loop_invariant
// CHECK:       [[CST_0:%.*]] = "tf.Const"
// CHECK-NEXT:  [[CST_1:%.*]] = "tf.Const"
// CHECK:       [[RES_1:%.*]] = "tf.Add"([[CST_1]], [[CST_0]])
// CHECK:       [[RES_2:%.*]] = "tf.Mul"([[RES_1]], [[CST_1]])
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0([[ARG_2:%[a-zA-Z0-9_]+]]
// CHECK-SAME:  [[ARG_3:%[a-zA-Z0-9_]+]]: tensor<i32>)
// CHECK-NEXT:  [[RES_3:%.*]] = "tf.AddV2"([[ARG_2]], [[RES_1]])
// CHECK-NEXT:  [[RES_4:%.*]] = "tf.Div"([[ARG_3]], [[RES_2]])
// CHECK:       "tf.Yield"([[RES_3]], [[RES_4]])
func.func @hoist_loop_invariant(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %1 = "tf.Add"(%cst_1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// Check that `tf.ReadVariableOp` is hoisted because the variable is readonly.
// The following ops that depend on `tf.ReadVariableOp` and satisfy hoisting
// conditions are also hoisted (i.e., `tf.Add` and `tf.Mul`).

// CHECK-LABEL: readvariableop_is_hoisted_if_readonly
// CHECK:       [[CST_0:%.*]] = "tf.Const"
// CHECK:       [[VAR:%.*]] = "tf.VarHandleOp"
// CHECK:       [[CST_1:%.*]] = "tf.Const"
// CHECK:       [[VAR_VAL:%.*]] = "tf.ReadVariableOp"
// CHECK:       [[RES_1:%.*]] = "tf.Add"([[VAR_VAL]], [[CST_0]])
// CHECK:       [[RES_2:%.*]] = "tf.Mul"([[RES_1]], [[CST_1]])
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0([[ARG_2:%[a-zA-Z0-9_]+]]
// CHECK-SAME:  [[ARG_3:%[a-zA-Z0-9_]+]]: tensor<i32>)
// CHECK-NEXT:  [[RES_3:%.*]] = "tf.AddV2"([[ARG_2]], [[RES_1]])
// CHECK-NEXT:  [[RES_4:%.*]] = "tf.Div"([[ARG_3]], [[RES_2]])
// CHECK:       "tf.Yield"([[RES_3]], [[RES_4]])
func.func @readvariableop_is_hoisted_if_readonly(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %var1 = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %val_var1 = "tf.ReadVariableOp"(%var1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %1 = "tf.Add"(%val_var1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// Check that `tf.ReadVariableOp` is hoisted because the variable is readonly.
// The following ops that depend on `tf.ReadVariableOp` and satisfy hoisting
// conditions are also hoisted (i.e., `tf.Add` and `tf.Mul`).
// Another variable `%var2` which has a different resource handle is not
// readonly.

// CHECK-LABEL: readvariableop_is_hoisted_if_readonly2
// CHECK:       [[CST_0:%.*]] = "tf.Const"
// CHECK:       [[VAR_1:%.*]] = "tf.VarHandleOp"
// CHECK-SAME:  "shared_name_var1"
// CHECK:       [[VAR_2:%.*]] = "tf.VarHandleOp"
// CHECK-SAME:  "shared_name_var2"
// CHECK:       [[CST_1:%.*]] = "tf.Const"
// CHECK:       [[VAR_VAL:%.*]] = "tf.ReadVariableOp"([[VAR_1]])
// CHECK:       [[RES_1:%.*]] = "tf.Add"([[VAR_VAL]], [[CST_0]])
// CHECK:       [[RES_2:%.*]] = "tf.Mul"([[RES_1]], [[CST_1]])
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0([[ARG_2:%[a-zA-Z0-9_]+]]
// CHECK-SAME:  [[ARG_3:%[a-zA-Z0-9_]+]]: tensor<i32>)
// CHECK-NEXT:  [[RES_3:%.*]] = "tf.AddV2"([[ARG_2]], [[RES_1]])
// CHECK-NEXT:  [[RES_4:%.*]] = "tf.Div"([[ARG_3]], [[RES_2]])
// CHECK-NEXT:  "tf.AssignVariableOp"([[VAR_2]], [[RES_1]])
// CHECK:       "tf.Yield"([[RES_3]], [[RES_4]])
func.func @readvariableop_is_hoisted_if_readonly2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %var1 = "tf.VarHandleOp"() {container="", shared_name="shared_name_var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %var2 = "tf.VarHandleOp"() {container="", shared_name="shared_name_var2", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %val_var1 = "tf.ReadVariableOp"(%var1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %1 = "tf.Add"(%val_var1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.AssignVariableOp"(%var2, %1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// Check that `tf.ReadVariableOp` is not hoisted because the variable is not
// readonly. The following ops that depend on `tf.ReadVariableOp` are not
// hoisted either.

// CHECK-LABEL: readvariableop_is_not_hoisted_if_not_readonly
// CHECK:       [[CST_0:%.*]] = "tf.Const"
// CHECK:       [[VAR:%.*]] = "tf.VarHandleOp"
// CHECK:       [[CST_1:%.*]] = "tf.Const"
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0([[ARG_2:%[a-zA-Z0-9_]+]]
// CHECK-SAME:  [[ARG_3:%[a-zA-Z0-9_]+]]: tensor<i32>)
// CHECK:       [[VAR_VAL:%.*]] = "tf.ReadVariableOp"
// CHECK:       [[RES_1:%.*]] = "tf.Add"([[VAR_VAL]], [[CST_0]])
// CHECK:       [[RES_2:%.*]] = "tf.Mul"([[RES_1]], [[CST_1]])
// CHECK-NEXT:  [[RES_3:%.*]] = "tf.AddV2"([[ARG_2]], [[RES_1]])
// CHECK-NEXT:  [[RES_4:%.*]] = "tf.Div"([[ARG_3]], [[RES_2]])
// CHECK-NEXT:  "tf.AssignVariableOp"([[VAR]], [[RES_1]])
// CHECK:       "tf.Yield"([[RES_3]], [[RES_4]])
func.func @readvariableop_is_not_hoisted_if_not_readonly(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %var1 = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %val_var1 = "tf.ReadVariableOp"(%var1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %1 = "tf.Add"(%val_var1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.AssignVariableOp"(%var1, %1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// Check that `tf.ReadVariableOp` is not hoisted because the variable is not
// readonly - another variable `%var2` has the same resource handle which is not
// readonly.

// CHECK-LABEL: readvariableop_is_not_hoisted_if_not_readonly2
// CHECK:       [[CST_0:%.*]] = "tf.Const"
// CHECK:       [[VAR_1:%.*]] = "tf.VarHandleOp"
// CHECK-SAME:  "shared_name_var1"
// CHECK:       [[VAR_2:%.*]] = "tf.VarHandleOp"
// CHECK-SAME:  "shared_name_var1"
// CHECK:       [[CST_1:%.*]] = "tf.Const"
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0([[ARG_2:%[a-zA-Z0-9_]+]]
// CHECK-SAME:  [[ARG_3:%[a-zA-Z0-9_]+]]: tensor<i32>)
// CHECK:       [[VAR_VAL:%.*]] = "tf.ReadVariableOp"([[VAR_1]])
// CHECK:       [[RES_1:%.*]] = "tf.Add"([[VAR_VAL]], [[CST_0]])
// CHECK:       [[RES_2:%.*]] = "tf.Mul"([[RES_1]], [[CST_1]])
// CHECK-NEXT:  [[RES_3:%.*]] = "tf.AddV2"([[ARG_2]], [[RES_1]])
// CHECK-NEXT:  [[RES_4:%.*]] = "tf.Div"([[ARG_3]], [[RES_2]])
// CHECK-NEXT:  "tf.AssignVariableOp"([[VAR_2]], [[RES_1]])
// CHECK:       "tf.Yield"([[RES_3]], [[RES_4]])
func.func @readvariableop_is_not_hoisted_if_not_readonly2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %var1 = "tf.VarHandleOp"() {container="", shared_name="shared_name_var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %var2 = "tf.VarHandleOp"() {container="", shared_name="shared_name_var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %val_var1 = "tf.ReadVariableOp"(%var1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %1 = "tf.Add"(%val_var1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.AssignVariableOp"(%var2, %1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// Check that `tf.ReadVariableOp` is not hoisted because the function arguments
// contain a resource. The following ops that depend on `tf.ReadVariableOp`
// are not hoisted either.

// CHECK-LABEL: readvariableop_not_hoisted_if_input_has_resource
// CHECK:       tf.VarHandleOp
// CHECK:       tf.WhileRegion
// CHECK:       ^bb0
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:       tf.ReadVariableOp
func.func @readvariableop_not_hoisted_if_input_has_resource(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>, tensor<i32>) {
  %cst_0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %var1 = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %cst_1 = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    %val_var1 = "tf.ReadVariableOp"(%var1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %1 = "tf.Add"(%val_var1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.Mul"(%1, %cst_1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.AddV2"(%arg3, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tf.Div"(%arg4, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.Yield"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: globaliterid_not_hoisted
// CHECK:       tf.WhileRegion
// CHECK:         tf.GlobalIterId
func.func @globaliterid_not_hoisted(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = "tf.GlobalIterId"() : () -> tensor<i64>
    "tf.Yield"(%arg2, %arg3) : (tensor<i32>, tensor<i32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0#0, %0#1 : tensor<i32>, tensor<i32>
}
