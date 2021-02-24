// RUN: tf-opt %s -tf-region-control-flow-to-functional -split-input-file | FileCheck %s

// Simple IfRegion
// CHECK: func private @test_else_name(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @test_then_name(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
func @testSimple(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"
  // CHECK-SAME: _attr0 = false
  // CHECK-NOT: attr1
  // CHECK-SAME: else_branch = @test_else_name
  // CHECK-SAME: then_branch = @test_then_name
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true, _attr0 = false, attr1 = "hello", _then_func_name = "test_then_name", _else_func_name = "test_else_name"} :  (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Simple IfRegion with empty branch names
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
func @testSimpleEmptyBranchNames(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"
  // CHECK-SAME: _attr0 = false
  // CHECK-NOT: attr1
  // CHECK-SAME: else_branch = @tf.IfRegion_else
  // CHECK-SAME: then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true, _attr0 = false, attr1 = "hello", _then_func_name = "", _else_func_name = ""} :  (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Use if condition inside the regions
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT: "tf.Select"(%arg0, %arg2, %arg3)
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT: "tf.Select"(%arg0, %arg1, %arg2)
func @testIfCondition(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.Add"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %1 = "tf.Mul"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %2 = "tf.Div"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %3 = "tf.IfRegion"(%arg0) ({
     %4 = "tf.Select"(%arg0, %0, %1) : (tensor<i1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%4) : (tensor<2xf32>) -> ()
    }, {
     %5 = "tf.Select"(%arg0, %1, %2):  (tensor<i1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%5) : (tensor<2xf32>) -> ()
    }) { is_stateless = true} : (tensor<i1>) -> tensor<2xf32>
   return %3 : tensor<2xf32>
}

// -----

// Constant sinking for IfRegion

// CHECK: func private @tf.IfRegion_else() -> tensor<2xf32>
// CHECK-NEXT: constant dense<1.0
// CHECK: func private @tf.IfRegion_then() -> tensor<2xf32>
// CHECK-NEXT: constant dense<0.0
func @testIfConstant(%arg0: tensor<i1>) -> tensor<2xf32> {
  %cst_zero = constant dense<0.0> : tensor<2xf32>
  // CHECK: "tf.If"(%arg0) {else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
     "tf.Yield"(%cst_zero) : (tensor<2xf32>) -> ()
    }, {
     %cst_one = constant dense<1.0> : tensor<2xf32>
     "tf.Yield"(%cst_one) : (tensor<2xf32>) -> ()
    }) { is_stateless = true} : (tensor<i1>) -> tensor<2xf32>
   return %0 : tensor<2xf32>
}

// -----

// Nested IfRegions
// CHECK: func private @tf.IfRegion1_else
// CHECK-NEXT: "tf.Acos"
// CHECK-NEXT: "tf.Abs"

// CHECK: func private @tf.IfRegion1_then
// CHECK-NEXT: "tf.LogicalNot"
// CHECK-NEXT: "tf.Asin"
// CHECK-NEXT: "tf.If"({{.+}}) {else_branch = @tf.IfRegion_else, {{.+}} then_branch = @tf.IfRegion_then}

// CHECK: func private @tf.IfRegion_else
// CHECK-NEXT: "tf.Neg"
// CHECK: func private @tf.IfRegion_then
// CHECK-NEXT: "tf.Abs"

func @testNested(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"({{.+}}) {else_branch = @tf.IfRegion1_else, {{.+}} then_branch = @tf.IfRegion1_then}
  %0 = "tf.IfRegion"(%arg0) ({
    // Outer Then
    %cond = "tf.LogicalNot"(%arg0) : (tensor<i1>) -> tensor<i1>
    %asin = "tf.Asin"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>

    // nested IfRegion
    %1 = "tf.IfRegion"(%cond) ({
        %2 = "tf.Abs"(%asin) : (tensor<*xf32>) -> tensor<*xf32>
        "tf.Yield"(%2) : (tensor<*xf32>) -> ()
      }, {
        %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
        "tf.Yield"(%2) : (tensor<*xf32>) -> ()
      }) { is_stateless = true } :  (tensor<i1>) -> tensor<*xf32>

    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    // Outer Else
    %acos = "tf.Acos"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.Abs"(%acos) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%3) : (tensor<*xf32>) -> ()
    }) { is_stateless = true } :  (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Match existing function->Region pattern (simple) for IfRegion
func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"({{.+}}) {else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ( {
    %1 = call @testIf1Then(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
  },  {
    %1 = call @testIf1Else(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
 }) {is_stateless = false} : (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Match existing function->Region pattern (with casts) for IfRegion

func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tf.If"({{.+}}) {else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ( {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<*xf32>
    %2 = call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<*xf32>
    %2 = call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// Match existing function->Region pattern (with multiple casts) for IfRegion

func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tf.If"({{.+}}) {else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ( {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<?xf32>
    %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<?xf32>) -> tensor<*xf32>
    %3 = call @testIf1Then(%2) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%3) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<?xf32>
    %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<?xf32>) -> tensor<*xf32>
    %3 = call @testIf1Else(%2) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%3) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// Do not skip extern incompatible cast for trivial transform.

func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func @testIfExternIncompatibleCastTrivialTransform(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<2xf32> {
  // CHECK: %[[CAST:.*]] = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
  // CHECK: "tf.If"(%arg0, %[[CAST]]) {else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
  %0 = "tf.IfRegion"(%arg0) ( {
    %2 = call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %2 = call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// Do not skip incompatible cast for trivial transform.

// CHECK: func private @tf.IfRegion_else(%arg0: tensor<2xi64>) -> tensor<*xf32>
// CHECK-NEXT:    "tf.Cast"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<2xi64>) -> tensor<*xf32>
// CHECK-NEXT:    "tf.Cast"
func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func @testIfIncompatibleCastTrivialTransform(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<2xf32> {
  // CHECK: "tf.If"(%arg0, %arg1) {else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then}
  %0 = "tf.IfRegion"(%arg0) ( {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
    %2 = call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
    %2 = call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// No inputs, some outputs for IfRegion
// CHECK: func private @tf.IfRegion_else() -> tensor<2xf32>
// CHECK-NEXT:    constant dense<1.000000e+00>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then() -> tensor<2xf32>
// CHECK-NEXT:   constant dense<0.000000e+00>
// CHECK-NEXT:   "tf.Abs"
func @testSimple(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
    %cst_zero = constant dense<0.0> : tensor<2xf32>
    %1 = "tf.Abs"(%cst_zero) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%1) : (tensor<2xf32>) -> ()
    }, {
    %cst_one = constant dense<1.0> : tensor<2xf32>
    %2 = "tf.Neg"(%cst_one) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%2) : (tensor<2xf32>) -> ()
    }) { is_stateless = true } :  (tensor<i1>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// No outputs, some inputs for IfRegion
//
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>)
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>)
// CHECK-NEXT:   "tf.Abs"
func private @printer(tensor<*xf32>) -> ()
func @testNoOutputs(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> () {
  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    call @printer(%1) : (tensor<*xf32>) -> ()
    "tf.Yield"() : () -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    call @printer(%2) : (tensor<*xf32>) -> ()
    "tf.Yield"() : () -> ()
    }) { is_stateless = false } :  (tensor<i1>) -> ()
  return
}

// -----
// Check ToBool folding for IfRegion
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
// CHECK-LABEL: @testToBoolFold
func @testToBoolFold(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: "tf.If"(%arg0, %arg1)
  // CHECK-SAME: else_branch = @tf.IfRegion_else
  // CHECK-SAME: then_branch = @tf.IfRegion_then
  %tobool = "tf.ToBool"(%arg0) : (tensor<i32>) -> tensor<i1>
  %0 = "tf.IfRegion"(%tobool) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true} :  (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Simple WhileRegion
// CHECK: func private @tf.WhileRegion_body{{.+}}
// CHECK: "tf.Add"
// CHECK: constant dense<1>
// CHECK: "tf.Sub"
// CHECK:func private @tf.WhileRegion_cond{{.+}}
// CHECK: constant dense<0>
// CHECK: "tf.NotEqual"
// CHECK-LABEL: testValidWhileRegion
func @testValidWhileRegion(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1)
  // CHECK-SAME: _attr0 = false
  // CHECK-NOT: attr1
  // CHECK-SAME: body = @tf.WhileRegion_body
  // CHECK-SAME: cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false, _attr0 = false, attr1 = "hello"} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with type mismatch
// CHECK: func private @tf.WhileRegion_body{{.+}}
// CHECK: "tf.Add"
// CHECK: constant dense<1>
// CHECK: "tf.Sub"
// CHECK:func private @tf.WhileRegion_cond{{.+}}
// CHECK: constant dense<0>
// CHECK: "tf.NotEqual"
// CHECK-LABEL: testWhileRegionTypeMismatch
func @testWhileRegionTypeMismatch(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
      %zero = constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %one = constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with constant sinking
// CHECK: func private @tf.WhileRegion_body{{.+}}
// CHECK: constant dense<1>
// CHECK: "tf.Add"
// CHECK: "tf.Sub"
// CHECK:func private @tf.WhileRegion_cond{{.+}}
// CHECK: constant dense<0>
// CHECK: "tf.NotEqual"
// CHECK-LABEL: testWhileRegionConstantSink
func @testWhileRegionConstantSink(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %zero = constant dense<0> : tensor<i32>
  %one = constant dense<1> : tensor<i32>
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with implicitly captured extern value in cond
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: "tf.Add"
// CHECK: constant dense<1>
// CHECK: "tf.Sub"
// CHECK: return %{{.+}}, %{{.+}}, %arg2 : tensor<*xf32>, tensor<i32>, tensor<i32>
// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: "tf.NotEqual"(%arg1, %arg2)
// CHECK-LABEL: testWhileRegionExternInCond
func @testWhileRegionExternInCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = constant dense<4> : tensor<i32>
  %limit = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}}) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %ne = "tf.NotEqual"(%carg1, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with implicitly captured extern value in body
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: %0 = "tf.Add"(%arg0, %arg0)
// CHECK: %1 = "tf.Sub"(%arg1, %arg2)
// CHECK: return %0, %1, %arg2

// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: constant dense<0>
// CHECK: "tf.NotEqual"

// CHECK-LABEL: testWhileRegionExternInBody
func @testWhileRegionExternInBody(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %zero = constant dense<0> : tensor<i32>
  %cst = constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}}) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %sub = "tf.Sub"(%barg1, %stride) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with implicitly captured extern value in cond and body
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
// CHECK: return %{{.+}}, %{{.+}}, %arg2, %arg3
// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
// CHECK-LABEL: testWhileRegionExternInBodyAndCond
func @testWhileRegionExternInBodyAndCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %cst1 = constant dense<44> : tensor<i32>
  %limit = "tf.Add"(%arg2, %cst1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:4 = "tf.While"(%arg0, %arg1, %{{.+}}, %{{.+}}) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %ne = "tf.NotEqual"(%carg1, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %sub = "tf.Sub"(%barg1, %stride) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with same value implicitly captured in cond and body
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: return %{{.+}}, %{{.+}}, %arg2
// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK-LABEL: testWhileRegionSameExternInBodyAndCond
func @testWhileRegionSameExternInBodyAndCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}}) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %ne = "tf.NotEqual"(%carg1, %stride) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %sub = "tf.Sub"(%barg1, %stride) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Simple trivially transformable while
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivial
func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func @testWhileRegionTrivial(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Trivially transformable with casts
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivialCasts
func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
func @testWhileRegionTrivialCasts(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_cast = "tf.Cast"(%carg0) : (tensor<*xf32>) -> tensor<4xf32>
        %cond = call @while_cond(%cond_cast, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy_cast = "tf.Cast"(%barg0) : (tensor<*xf32>) -> tensor<4xf32>
        %bdy:2 = call @while_body(%bdy_cast, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Trivially transformable with multiple casts
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivialMultipleCasts
func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
func @testWhileRegionTrivialMultipleCasts(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_cast0 = "tf.Cast"(%carg0) : (tensor<*xf32>) -> tensor<?xf32>
        %cond_cast1 = "tf.Cast"(%cond_cast0) : (tensor<?xf32>) -> tensor<4xf32>
        %cond = call @while_cond(%cond_cast1, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy_cast0 = "tf.Cast"(%barg0) : (tensor<*xf32>) -> tensor<?xf32>
        %bdy_cast1 = "tf.Cast"(%bdy_cast0) : (tensor<?xf32>) -> tensor<4xf32>
        %bdy:2 = call @while_body(%bdy_cast1, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Almost trivially transformable with incompatible cast
// CHECK: func private @tf.WhileRegion_body
// CHECK-NEXT:    "tf.Cast"
// CHECK: func private @tf.WhileRegion_cond
// CHECK-NEXT:    "tf.Cast"
// CHECK-LABEL: testWhileRegionIncompatibleCast
func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xi64>, tensor<i32>)
func @testWhileRegionIncompatibleCast(%arg0 : tensor<*xi64>, %arg1 : tensor<i32>) -> tensor<*xi64> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xi64>, %carg1: tensor<i32>):
        %cond_cast = "tf.Cast"(%carg0) : (tensor<*xi64>) -> tensor<4xf32>
        %cond = call @while_cond(%cond_cast, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xi64>, %barg1: tensor<i32>):
        %bdy_cast = "tf.Cast"(%barg0) : (tensor<*xi64>) -> tensor<4xf32>
        %bdy:2 = call @while_body(%bdy_cast, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xi64>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xi64>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xi64>, tensor<i32>) -> (tensor<*xi64>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xi64>
}

// -----

// Almost trivially transformable with extern values
// CHECK: func private @tf.WhileRegion_body
// CHECK: call @while_body
// CHECK: func private @tf.WhileRegion_cond
// CHECK: call @while_cond
// CHECK-LABEL: testWhileRegionExtern
func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>)
func @testWhileRegionExtern(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %ext = "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}}) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = call @while_body(%barg0, %barg1, %ext) : (tensor<*xf32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Almost trivially transformable, mismatching block arguments
// CHECK: func private @tf.WhileRegion_body
// CHECK: call @while_body
// CHECK: func private @tf.WhileRegion_cond
// CHECK: call @while_cond
// CHECK-LABEL: testWhileRegionBlockArgMismatch
func private @while_cond(%arg0 : tensor<i32>, %arg1 : tensor<*xf32>) -> tensor<i1>
func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func @testWhileRegionBlockArgMismatch(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = call @while_cond(%carg1, %carg0) : (tensor<i32>, tensor<*xf32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Simple trivially transformable while with ToBool
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivial
func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i32>
func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func @testWhileRegionTrivial(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) {body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_i32 = call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i32>
        %cond = "tf.ToBool"(%cond_i32) : (tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  return %0#0 : tensor<*xf32>
}

// -----

// Test tf.IfRegion device is preserved.
// CHECK-LABEL: func @testIfRegionDevice
func @testIfRegionDevice(%arg0: tensor<i1>) {
  "tf.IfRegion"(%arg0) ({
    "tf.Yield"() : () -> ()
  }, {
    "tf.Yield"() : () -> ()
  }) {is_stateless = false, device = "/device:CPU:0"} : (tensor<i1>) -> ()

  // CHECK: "tf.If"
  // CHECK-SAME: device = "/device:CPU:0"
  return
}

// -----

// Test tf.WhileRegion device is preserved.
// CHECK-LABEL: func @testWhileRegionDevice
func @testWhileRegionDevice() {
  "tf.WhileRegion"() ( {
    %0 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    "tf.Yield"(%0) : (tensor<i1>) -> ()
  }, {
    "tf.Yield"() : () -> ()
  }) {is_stateless = false, device = "/device:CPU:0"} : () -> ()

  // CHECK: "tf.While"
  // CHECK-SAME: device = "/device:CPU:0"
  return
}
