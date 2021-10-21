// RUN: tf-opt %s -tf-freeze-variables-test-pass -verify-diagnostics -split-input-file | FileCheck %s

// Test case: Basic freezing.

module {
  // CHECK: func @main()
  func @main() -> tensor<0xf32> {
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> tensor<0xf32>
    // CHECK-NOT: "tf.VarHandleOp"
    return %val : tensor<0xf32>
  }
}

// -----

// Variable is mutated, nothing should be removed.
module {
  func @f() {
    %cst = "tf.Const"() { value = dense<1.0> : tensor<0xf32> } : () -> tensor<0xf32>
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    "tf.AssignVariableOp"(%handle, %cst) : (tensor<!tf_type.resource<tensor<0xf32>>>, tensor<0xf32>) -> ()
    // CHECK: "tf.VarHandleOp"
    return
  }
}

// -----

// Read and write usage.
module {
  func @f() {
    %cst = "tf.Const"() { value = dense<1.0> : tensor<0xf32> } : () -> tensor<0xf32>
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> tensor<0xf32>
    %0 = "tf.AddV2"(%val, %val) : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
    "tf.AssignVariableOp"(%handle, %0) : (tensor<!tf_type.resource<tensor<0xf32>>>, tensor<0xf32>) -> ()
    // CHECK: "tf.VarHandleOp"
    return
  }
}

// -----

// Test mutation detection propagates across function calls.

module {
  func @f() -> tensor<0xf32> {
    // CHECK: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.PartitionedCall"(%handle) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32>
  func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32>
  func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %c0 = "tf.Const"() { value = dense<1.0> : tensor<0xf32> } : () -> tensor<0xf32>
    "tf.AssignVariableOp"(%arg0, %c0) : (tensor<*x!tf_type.resource>, tensor<0xf32>) -> ()
    return %c0 : tensor<0xf32>
  }
}

// -----

// Test immutable detection propagates across function calls.

module {
  func @f() -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.PartitionedCall"(%handle) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee() -> tensor<0xf32>
  func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee_callee() -> tensor<0xf32>
  func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }
}


// -----

// Test mutable detection propagates across function calls. No freezing

module {
  func @f() -> tensor<0xf32> {
    // CHECK: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.PartitionedCall"(%handle) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32>
  func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32>
  func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource>) -> tensor<0xf32>
    %0 = "tf.AddV2"(%val, %val) : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
    "tf.AssignVariableOp"(%arg0, %0) : (tensor<*x!tf_type.resource>, tensor<0xf32>) -> ()
    return %0 : tensor<0xf32>
  }
}


// -----

// Test immutable in If condition.

module {
  // CHECK: func private @testIfThen(%arg0: tensor<*xf32>)
  func private @testIfThen(%arg0:tensor<*xf32>, %arg1:tensor<*x!tf_type.resource>) -> tensor<*xf32> {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %0 = "tf.AddV2"(%val, %arg0) : (tensor<0xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

  // CHECK: func private @testIfElse(%arg0: tensor<*xf32>) -> tensor<*xf32>
  func private @testIfElse(%arg0:tensor<*xf32>, %arg1:tensor<*x!tf_type.resource>) -> tensor<*xf32> {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %0 = "tf.Mul"(%val, %arg0) : (tensor<0xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

  func @f(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %1 = "tf.If"(%arg0, %arg1, %handle) {
      then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
    } : (tensor<i1>, tensor<2xf32>, tensor<!tf_type.resource<tensor<0xf32>>>) -> tensor<2xf32>

    return %1 : tensor<2xf32>
  }
}

// -----

// Test mutable in If condition.

module {
  // CHECK: @testIfThen(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf_type.resource>)
  func private @testIfThen(%arg0:tensor<*xf32>, %arg1:tensor<*x!tf_type.resource>) -> tensor<*xf32> {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %0 = "tf.AddV2"(%val, %arg0) : (tensor<0xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tf.AssignVariableOp"(%arg1, %0) : (tensor<*x!tf_type.resource>, tensor<*xf32>) -> ()
    return %0 : tensor<*xf32>
  }

  // CHECK: @testIfElse(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf_type.resource>)
  func private @testIfElse(%arg0:tensor<*xf32>, %arg1:tensor<*x!tf_type.resource>) -> tensor<*xf32> {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %0 = "tf.Mul"(%val, %arg0) : (tensor<0xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

  func @f(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %1 = "tf.If"(%arg0, %arg1, %handle) {
      then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
    } : (tensor<i1>, tensor<2xf32>, tensor<!tf_type.resource<tensor<0xf32>>>) -> tensor<2xf32>

    return %1 : tensor<2xf32>
  }
}

// -----

// Test for immutable case on while loop.
module {
  // CHECK-LABEL: @cond(%arg0: tensor<0xf32>) -> tensor<i1>
  func private @cond(%arg0:tensor<0xf32>, %arg1:tensor<*x!tf_type.resource>) -> tensor<i1> {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %cst = arith.constant dense<-2.0>: tensor<0xf32>
    %cst_1 = arith.constant dense<0> : tensor<i32>
    %greater = "tf.Greater"(%val, %cst) : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xi1>
    %res = "tf.Any"(%greater, %cst_1) : (tensor<0xi1>, tensor<i32>) -> (tensor<i1>)
    return %res : tensor<i1>
  }

  // CHECK-LABEL: @body(%arg0: tensor<0xf32>) -> tensor<0xf32>
  func private @body(%arg0:tensor<0xf32>, %arg1:tensor<*x!tf_type.resource>) -> (tensor<0xf32>, tensor<*x!tf_type.resource>) {
    %val = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    %0 = "tf.Mul"(%val, %arg0) : (tensor<0xf32>, tensor<0xf32>) -> tensor<0xf32>
    return %0, %arg1 : tensor<0xf32>, tensor<*x!tf_type.resource>
  }

  // CHECK-LABEL: @f
  func @f(%arg0: tensor<0xf32>) -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %0, %1 = "tf.While"(%arg0, %handle) {cond = @cond, body = @body, is_stateless = false} : (tensor<0xf32>, tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>, tensor<!tf_type.resource<tensor<*xf32>>>)
    return %0 : tensor<0xf32>
  }
}

// -----

// Test immutable detection propagates across function calls, with recursion.

module {
  // CHECK-LABEL: @f()
  func @f() -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.PartitionedCall"(%handle) {config = "", config_proto = "", executor_type = "", f = @f_callee} : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee() -> tensor<0xf32>
  func private @f_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %val = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %val : tensor<0xf32>
  }

  // CHECK: func private @f_callee_callee() -> tensor<0xf32>
  func private @f_callee_callee(%arg0: tensor<*x!tf_type.resource>) -> tensor<0xf32> {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @f_callee_callee} : (tensor<*x!tf_type.resource>) -> (tensor<0xf32>)
    return %0 : tensor<0xf32>
  }
}

// -----

// Test If Region immutable case.

module {
  func @f(%arg0: tensor<i1>) -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %0 = "tf.IfRegion"(%arg0) ( {
      %1 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
      "tf.Yield"(%1) : (tensor<0xf32>) -> ()
     },  {
      %2 = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
      "tf.Yield"(%2) : (tensor<0xf32>) -> ()
    }) {is_stateless = true} : (tensor<i1>) -> (tensor<0xf32>)
    return %0 : tensor<0xf32>
  }
}

// -----

// Test While region immutable case.
module {
  func @f(%arg0: tensor<i32>) -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %0 = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
    %1:2 = "tf.WhileRegion"(%arg0, %0) ( {
      ^bb0(%carg0: tensor<i32>, %carg1: tensor<0xf32>):
         %limit = arith.constant dense<5> : tensor<i32>
         %cond = "tf.NotEqual"(%carg0, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
         "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },  {
      ^bb0(%barg0: tensor<i32>, %barg1: tensor<0xf32>):
        %2 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
        "tf.Yield"(%barg0, %2) : (tensor<i32>, tensor<0xf32>) -> ()
    }) {is_stateless = true} : (tensor<i32>, tensor<0xf32>) -> (tensor<i32>, tensor<0xf32>)
    return %1#1 : tensor<0xf32>
  }
}


// -----

// Test While region immutable case.
module {
  func @f(%arg0: tensor<i32>) -> tensor<0xf32> {
    // CHECK-NOT: "tf.VarHandleOp"
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %0 = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
    %1:3 = "tf.WhileRegion"(%arg0, %0, %handle) ( {
      // CHECK: ^bb0(%arg1: tensor<i32>, %arg2: tensor<0xf32>)
      ^bb0(%carg0: tensor<i32>, %carg1: tensor<0xf32>, %carg2: tensor<!tf_type.resource<tensor<0xf32>>>):
         %limit = arith.constant dense<5> : tensor<i32>
         %cond = "tf.NotEqual"(%carg0, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
         "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },  {
      // CHECK: ^bb0(%arg1: tensor<i32>, %arg2: tensor<0xf32>)
      ^bb0(%barg0: tensor<i32>, %barg1: tensor<0xf32>, %barg2: tensor<!tf_type.resource<tensor<0xf32>>>):
        %2 = "tf.ReadVariableOp"(%barg2) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
        "tf.Yield"(%barg0, %2, %barg2) : (tensor<i32>, tensor<0xf32>, tensor<!tf_type.resource<tensor<0xf32>>>) -> ()
    }) {is_stateless = true} : (tensor<i32>, tensor<0xf32>, tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<i32>, tensor<0xf32>, tensor<!tf_type.resource<tensor<0xf32>>>)
    return %1#1 : tensor<0xf32>
  }
}


// -----

// Make sure multiple entry points is still captured as mutation.
module {
  func @f() {
    %cst = "tf.Const"() { value = dense<1.0> : tensor<0xf32> } : () -> tensor<0xf32>
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    "tf.AssignVariableOp"(%handle, %cst) : (tensor<!tf_type.resource<tensor<0xf32>>>, tensor<0xf32>) -> ()
    // CHECK: "tf.VarHandleOp"
    return
  }

  func @f2() -> tensor<0xf32> {
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %0 = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> (tensor<0xf32>)
    // CHECK: "tf.VarHandleOp"
    return %0 : tensor<0xf32>
  }
}

// -----

// Make sure Session init function is not considered mutation and
// initialization is removed.
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {
  "tf_saved_model.session_initializer"() {initializers = [@Init]} : () -> ()
  func @Init() attributes {tf_saved_model.exported_names = ["Init"]} {
    // CHECK-NOT: "tf.VarHandleOp"
    %cst = "tf.Const"() { value = dense<1.0> : tensor<0xf32> } : () -> tensor<0xf32>
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    "tf.AssignVariableOp"(%handle, %cst) : (tensor<!tf_type.resource<tensor<0xf32>>>, tensor<0xf32>) -> ()
    return
  }

  // CHECK-LABEL: func @main()
  func @main() -> (tensor<0xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %handle = "tf.VarHandleOp"() {container="", shared_name="var1", device = "/job:worker/replica:0/task:1/device:CPU:0"} : () -> tensor<!tf_type.resource<tensor<0xf32>>>
    %val = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<0xf32>>>) -> tensor<0xf32>
    // CHECK-NOT: "tf.VarHandleOp"
    return %val : tensor<0xf32>
  }
}
