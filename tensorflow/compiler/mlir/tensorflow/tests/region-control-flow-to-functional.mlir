// RUN: tf-opt %s -tf-region-control-flow-to-functional -split-input-file | FileCheck %s

// Simple IfRegion
// CHECK: func private @test_else_name(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @test_then_name(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
func.func @testSimple(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"
  // CHECK-NOT: attr1
  // CHECK-SAME: else_branch = @test_else_name
  // CHECK-SAME: then_branch = @test_then_name
  // CHECK-SAME: _attr0 = false
  // CHECK-SAME: _xla_propagate_compile_time_consts = true
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true, _attr0 = false, attr1 = "hello", _then_func_name = "test_then_name", _else_func_name = "test_else_name"} :  (tensor<i1>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Simple IfRegion with empty branch names
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
func.func @testSimpleEmptyBranchNames(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"
  // CHECK-NOT: attr1
  // CHECK-SAME: else_branch = @tf.IfRegion_else
  // CHECK-SAME: then_branch = @tf.IfRegion_then
  // CHECK-SAME: _attr0 = false
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true, _attr0 = false, attr1 = "hello", _then_func_name = "", _else_func_name = ""} :  (tensor<i1>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Use if condition inside the regions
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT: "tf.Select"(%arg0, %arg2, %arg3)
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT: "tf.Select"(%arg0, %arg1, %arg2)
func.func @testIfCondition(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
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
   func.return %3 : tensor<2xf32>
}

// -----

// Constant sinking for IfRegion

// CHECK: func private @tf.IfRegion_else() -> tensor<2xf32>
// CHECK-NEXT: constant dense<1.0
// CHECK: func private @tf.IfRegion_then() -> tensor<2xf32>
// CHECK-NEXT: constant dense<0.0
func.func @testIfConstant(%arg0: tensor<i1>) -> tensor<2xf32> {
  %cst_zero = arith.constant dense<0.0> : tensor<2xf32>
  // CHECK: "tf.If"(%arg0) <{else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
     "tf.Yield"(%cst_zero) : (tensor<2xf32>) -> ()
    }, {
     %cst_one = arith.constant dense<1.0> : tensor<2xf32>
     "tf.Yield"(%cst_one) : (tensor<2xf32>) -> ()
    }) { is_stateless = true} : (tensor<i1>) -> tensor<2xf32>
   func.return %0 : tensor<2xf32>
}

// -----

// Nested IfRegions
// CHECK: func private @tf.IfRegion1_else
// CHECK-NEXT: "tf.Acos"
// CHECK-NEXT: "tf.Abs"

// CHECK: func private @tf.IfRegion1_then
// CHECK-NEXT: "tf.LogicalNot"
// CHECK-NEXT: "tf.Asin"
// CHECK-NEXT: "tf.If"({{.+}}) <{else_branch = @tf.IfRegion_else, {{.+}} then_branch = @tf.IfRegion_then}

// CHECK: func private @tf.IfRegion_else
// CHECK-NEXT: "tf.Neg"
// CHECK: func private @tf.IfRegion_then
// CHECK-NEXT: "tf.Abs"

func.func @testNested(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"({{.+}}) <{else_branch = @tf.IfRegion1_else, {{.+}} then_branch = @tf.IfRegion1_then}
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
  func.return %0 : tensor<*xf32>
}

// -----

// Match existing function->Region pattern (simple) for IfRegion
func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func.func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"({{.+}}) <{else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = func.call @testIf1Then(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
  },  {
    %1 = func.call @testIf1Else(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
 }) {is_stateless = false} : (tensor<i1>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Match existing function->Region pattern (with casts) for IfRegion

func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func.func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tf.If"({{.+}}) <{else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<*xf32>
    %2 = func.call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<*xf32>
    %2 = func.call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Match existing function->Region pattern (with multiple casts) for IfRegion

func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func.func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: "tf.If"({{.+}}) <{else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<?xf32>
    %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<?xf32>) -> tensor<*xf32>
    %3 = func.call @testIf1Then(%2) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%3) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xf32>) -> tensor<?xf32>
    %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<?xf32>) -> tensor<*xf32>
    %3 = func.call @testIf1Else(%2) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%3) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Do not skip extern incompatible cast for trivial transform.

func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func.func @testIfExternIncompatibleCastTrivialTransform(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<2xf32> {
  // CHECK: %[[CAST:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<2xi64>) -> tensor<*xf32>
  // CHECK: "tf.If"(%arg0, %[[CAST]]) <{else_branch = @testIf1Else, {{.+}} then_branch = @testIf1Then}
  %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
  %0 = "tf.IfRegion"(%arg0) ({
    %2 = func.call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %2 = func.call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Do not skip incompatible cast for trivial transform.

// CHECK: func private @tf.IfRegion_else(%arg0: tensor<2xi64>) -> tensor<*xf32>
// CHECK-NEXT:    "tf.Cast"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<2xi64>) -> tensor<*xf32>
// CHECK-NEXT:    "tf.Cast"
func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>
func.func @testIfIncompatibleCastTrivialTransform(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<2xf32> {
  // CHECK: "tf.If"(%arg0, %arg1) <{else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then}
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
    %2 = func.call @testIf1Then(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  },  {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2xi64>) -> tensor<*xf32>
    %2 = func.call @testIf1Else(%1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// No inputs, some outputs for IfRegion
// CHECK: func private @tf.IfRegion_else() -> tensor<2xf32>
// CHECK-NEXT:    constant dense<1.000000e+00>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then() -> tensor<2xf32>
// CHECK-NEXT:   constant dense<0.000000e+00>
// CHECK-NEXT:   "tf.Abs"
func.func @testSimple(%arg0: tensor<i1>) -> tensor<2xf32> {
  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
    %cst_zero = arith.constant dense<0.0> : tensor<2xf32>
    %1 = "tf.Abs"(%cst_zero) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%1) : (tensor<2xf32>) -> ()
    }, {
    %cst_one = arith.constant dense<1.0> : tensor<2xf32>
    %2 = "tf.Neg"(%cst_one) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%2) : (tensor<2xf32>) -> ()
    }) { is_stateless = true } :  (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// No outputs, some inputs for IfRegion
//
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>)
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>)
// CHECK-NEXT:   "tf.Abs"
func.func private @printer(tensor<*xf32>) -> ()
func.func @testNoOutputs(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> () {
  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    func.call @printer(%1) : (tensor<*xf32>) -> ()
    "tf.Yield"() : () -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    func.call @printer(%2) : (tensor<*xf32>) -> ()
    "tf.Yield"() : () -> ()
    }) { is_stateless = false } :  (tensor<i1>) -> ()
  func.return
}

// -----
// Check ToBool folding for IfRegion
// CHECK: func private @tf.IfRegion_else(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Neg"
// CHECK: func private @tf.IfRegion_then(%arg0: tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT:   "tf.Abs"
// CHECK-LABEL: @testToBoolFold
func.func @testToBoolFold(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
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
  func.return %0 : tensor<*xf32>
}

// -----

func.func private @branch_0(tensor<!tf_type.resource>) -> tensor<*xf32>
func.func private @branch_1(tensor<!tf_type.resource>) -> tensor<*xf32>

// CHECK: func private @tf.CaseRegion_branch1{{.*}}
// CHECK: call @branch_1
// CHECK: func private @tf.CaseRegion_branch0{{.*}}
// CHECK: call @branch_0
// CHECK-LABEL: func @testCase
func.func @testCase(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32> {
  // CHECK: [[Result:%.*]] = "tf.Case"(%arg0, %arg1)
  // CHECK-SAME: branches = [@tf.CaseRegion_branch0{{.*}}, @tf.CaseRegion_branch1{{.*}}]
  // CHECK-SAME: is_stateless = false
  %0 = "tf.CaseRegion"(%arg0) ({
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<!tf_type.resource>
    %2 = func.call @branch_0(%1) : (tensor<!tf_type.resource>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }, {
    %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<!tf_type.resource>
    %2 = func.call @branch_1(%1) : (tensor<!tf_type.resource>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<i32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
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
func.func @testValidWhileRegion(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1)
  // CHECK-NOT: attr1
  // CHECK-SAME: body = @tf.WhileRegion_body
  // CHECK-SAME: cond = @tf.WhileRegion_cond
  // CHECK-SAME: _attr0 = false
  // CHECK-SAME: _xla_propagate_compile_time_consts = true
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false, _attr0 = false, attr1 = "hello"} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
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
func.func @testWhileRegionTypeMismatch(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
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
func.func @testWhileRegionConstantSink(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %zero = arith.constant dense<0> : tensor<i32>
  %one = arith.constant dense<1> : tensor<i32>
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
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
  func.return %0#0 : tensor<*xf32>
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
func.func @testWhileRegionExternInCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = arith.constant dense<4> : tensor<i32>
  %limit = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}} <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
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
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
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
func.func @testWhileRegionExternInBody(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %zero = arith.constant dense<0> : tensor<i32>
  %cst = arith.constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}} <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
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
  func.return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with implicitly captured extern value in cond and body
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
// CHECK: return %{{.+}}, %{{.+}}, %arg2, %arg3
// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>)
// CHECK-LABEL: testWhileRegionExternInBodyAndCond
func.func @testWhileRegionExternInBodyAndCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = arith.constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %cst1 = arith.constant dense<44> : tensor<i32>
  %limit = "tf.Add"(%arg2, %cst1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:4 = "tf.While"(%arg0, %arg1, %{{.+}}, %{{.+}} <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
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
  func.return %0#0 : tensor<*xf32>
}

// -----

// WhileRegion with same value implicitly captured in cond and body
// CHECK: func private @tf.WhileRegion_body(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK: return %{{.+}}, %{{.+}}, %arg2
// CHECK: func private @tf.WhileRegion_cond(%arg0: tensor<*xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
// CHECK-LABEL: testWhileRegionSameExternInBodyAndCond
func.func @testWhileRegionSameExternInBodyAndCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<*xf32> {
  %cst = arith.constant dense<4> : tensor<i32>
  %stride = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}} <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
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
  func.return %0#0 : tensor<*xf32>
}

// -----

// Simple trivially transformable while
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivial
func.func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func.func @testWhileRegionTrivial(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = func.call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = func.call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Trivially transformable with casts
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivialCasts
func.func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
func.func @testWhileRegionTrivialCasts(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_cast = "tf.Cast"(%carg0) : (tensor<*xf32>) -> tensor<4xf32>
        %cond = func.call @while_cond(%cond_cast, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy_cast = "tf.Cast"(%barg0) : (tensor<*xf32>) -> tensor<4xf32>
        %bdy:2 = func.call @while_body(%bdy_cast, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Trivially transformable with multiple casts
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivialMultipleCasts
func.func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
func.func @testWhileRegionTrivialMultipleCasts(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_cast0 = "tf.Cast"(%carg0) : (tensor<*xf32>) -> tensor<?xf32>
        %cond_cast1 = "tf.Cast"(%cond_cast0) : (tensor<?xf32>) -> tensor<4xf32>
        %cond = func.call @while_cond(%cond_cast1, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy_cast0 = "tf.Cast"(%barg0) : (tensor<*xf32>) -> tensor<?xf32>
        %bdy_cast1 = "tf.Cast"(%bdy_cast0) : (tensor<?xf32>) -> tensor<4xf32>
        %bdy:2 = func.call @while_body(%bdy_cast1, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Almost trivially transformable with incompatible cast
// CHECK: func private @tf.WhileRegion_body
// CHECK-NEXT:    "tf.Cast"
// CHECK: func private @tf.WhileRegion_cond
// CHECK-NEXT:    "tf.Cast"
// CHECK-LABEL: testWhileRegionIncompatibleCast
func.func private @while_cond(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> (tensor<4xi64>, tensor<i32>)
func.func @testWhileRegionIncompatibleCast(%arg0 : tensor<*xi64>, %arg1 : tensor<i32>) -> tensor<*xi64> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xi64>, %carg1: tensor<i32>):
        %cond_cast = "tf.Cast"(%carg0) : (tensor<*xi64>) -> tensor<4xf32>
        %cond = func.call @while_cond(%cond_cast, %carg1) : (tensor<4xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xi64>, %barg1: tensor<i32>):
        %bdy_cast = "tf.Cast"(%barg0) : (tensor<*xi64>) -> tensor<4xf32>
        %bdy:2 = func.call @while_body(%bdy_cast, %barg1) : (tensor<4xf32>, tensor<i32>) -> (tensor<4xi64>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<4xi64>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xi64>, tensor<i32>) -> (tensor<*xi64>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xi64>
}

// -----

// Almost trivially transformable with extern values
// CHECK: func private @tf.WhileRegion_body
// CHECK: call @while_body
// CHECK: func private @tf.WhileRegion_cond
// CHECK: call @while_cond
// CHECK-LABEL: testWhileRegionExtern
func.func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2 : tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>)
func.func @testWhileRegionExtern(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %ext = "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: [[Result:%.*]]:3 = "tf.While"(%arg0, %arg1, %{{.+}} <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = func.call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = func.call @while_body(%barg0, %barg1, %ext) : (tensor<*xf32>, tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Almost trivially transformable, mismatching block arguments
// CHECK: func private @tf.WhileRegion_body
// CHECK: call @while_body
// CHECK: func private @tf.WhileRegion_cond
// CHECK: call @while_cond
// CHECK-LABEL: testWhileRegionBlockArgMismatch
func.func private @while_cond(%arg0 : tensor<i32>, %arg1 : tensor<*xf32>) -> tensor<i1>
func.func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func.func @testWhileRegionBlockArgMismatch(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @tf.WhileRegion_body, cond = @tf.WhileRegion_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond = func.call @while_cond(%carg1, %carg0) : (tensor<i32>, tensor<*xf32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = func.call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Simple trivially transformable while with ToBool
// CHECK: func private @while_cond
// CHECK: func private @while_body
// CHECK-LABEL: testWhileRegionTrivial
func.func private @while_cond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<i32>
func.func private @while_body(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
func.func @testWhileRegionTrivial(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1) <{body = @while_body, cond = @while_cond
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
        %cond_i32 = func.call @while_cond(%carg0, %carg1) : (tensor<*xf32>, tensor<i32>) -> tensor<i32>
        %cond = "tf.ToBool"(%cond_i32) : (tensor<i32>) -> tensor<i1>
        "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
        %bdy:2 = func.call @while_body(%barg0, %barg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
        "tf.Yield"(%bdy#0, %bdy#1) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Test tf.IfRegion device is preserved.
// CHECK-LABEL: func @testIfRegionDevice
func.func @testIfRegionDevice(%arg0: tensor<i1>) {
  "tf.IfRegion"(%arg0) ({
    "tf.Yield"() : () -> ()
  }, {
    "tf.Yield"() : () -> ()
  }) {is_stateless = false, device = "/device:CPU:0"} : (tensor<i1>) -> ()

  // CHECK: "tf.If"
  // CHECK-SAME: device = "/device:CPU:0"
  func.return
}

// -----

// Test tf.WhileRegion device is preserved.
// CHECK-LABEL: func @testWhileRegionDevice
func.func @testWhileRegionDevice() {
  "tf.WhileRegion"() ({
    %0 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    "tf.Yield"(%0) : (tensor<i1>) -> ()
  }, {
    "tf.Yield"() : () -> ()
  }) {is_stateless = false, device = "/device:CPU:0"} : () -> ()

  // CHECK: "tf.While"
  // CHECK-SAME: device = "/device:CPU:0"
  func.return
}

// -----

// CHECK-LABEL: @testOverrideIfRegionXlaPropageCompileTimeConsts
func.func @testOverrideIfRegionXlaPropageCompileTimeConsts(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"
  // CHECK-SAME: _xla_propagate_compile_time_consts = true
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) {is_stateless = true, _attr0 = false, attr1 = "hello", _then_func_name = "test_then_name", _else_func_name = "test_else_name", _xla_propagate_compile_time_consts = false} :  (tensor<i1>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Trivial WhileRegion_cond
// CHECK: func private @tf.WhileRegion_body{{.+}}
// CHECK: "tf.Add"
// CHECK: constant dense<1>
// CHECK: "tf.Sub"
// CHECK:func private @tf.WhileRegion_cond{{.+}}
// CHECK: "tf.ToBool"
// CHECK-LABEL: testValidWhileRegion
func.func @testValidWhileRegion(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: [[Result:%.*]]:2 = "tf.While"(%arg0, %arg1)
  // CHECK-NOT: attr1
  // CHECK-SAME: body = @tf.WhileRegion_body
  // CHECK-SAME: cond = @tf.WhileRegion_cond
  // CHECK-SAME: _attr0 = false
  // CHECK-SAME: _xla_propagate_compile_time_consts = true
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %cond = "tf.ToBool"(%carg1) : (tensor<i32>) -> tensor<i1>
      "tf.Yield"(%cond) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false, _attr0 = false, attr1 = "hello"} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return [[Result]]#0
  func.return %0#0 : tensor<*xf32>
}

// -----

// Condition with passthrough arguments
// CHECK: func private @tf.WhileRegion_body{{.+}}
// CHECK: func private @tf.WhileRegion_cond{{.+}}
// CHECK:   return {{[^,]*}} :
// CHECK-LABEL: testPassThroughCond
func.func @testPassThroughCond(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %cond = "tf.ToBool"(%carg1) : (tensor<i32>) -> tensor<i1>
      "tf.Yield"(%cond, %carg0, %carg1) : (tensor<i1>, tensor<*xf32>, tensor<i32>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false, _attr0 = false, attr1 = "hello"} : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  func.return %0#0 : tensor<*xf32>
}

// -----

func.func @init(%arg0: tensor<4xf32>) -> tensor<7xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<7xf32>
  return %0 : tensor<7xf32>
}
func.func @next(%arg0: tensor<7xf32>, %arg1: tensor<3xf32>) -> tensor<6xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<6xf32>
  return %0 : tensor<6xf32>
}
func.func @finalize(%arg0: tensor<6xf32>, %arg1: tensor<2xf32>) -> tensor<5xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<5xf32>
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: testGeneratorDatasetRegion
func.func @testGeneratorDatasetRegion(%arg0: tensor<4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<!tf_type.resource>, %arg3: tensor<2xf32>) {
  // CHECK: "tf.GeneratorDataset"
  // CHECK-DAG: @init
  // CHECK-DAG: @next
  // CHECK-DAG: @finalize
  // CHECK: return
  %0 = "tf.GeneratorDatasetRegion"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<4xf32>):
    %1 = func.call @init(%arg4) : (tensor<4xf32>) -> tensor<7xf32>
    "tf.Yield"(%1) : (tensor<7xf32>) -> ()
  }, {
  ^bb0(%arg4: tensor<7xf32>, %arg5: tensor<3xf32>):
    %1 = func.call @next(%arg4, %arg5) : (tensor<7xf32>, tensor<3xf32>) -> tensor<6xf32>
    "tf.Yield"(%1) : (tensor<6xf32>) -> ()
  }, {
  ^bb0(%arg4: tensor<6xf32>, %arg5: tensor<2xf32>):
    %1 = func.call @finalize(%arg4, %arg5) : (tensor<6xf32>, tensor<2xf32>) -> tensor<5xf32>
    "tf.Yield"(%1) : (tensor<5xf32>) -> ()
  }) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", metadata = "", operandSegmentSizes = array<i32: 1, 2, 1>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string]} : (tensor<4xf32>, tensor<3xf32>, tensor<!tf_type.resource>, tensor<2xf32>) -> tensor<!tf_type.variant>
  return
}

// -----

func.func @init(%arg0: tensor<4xf32>) -> tensor<7xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<7xf32>
  return %0 : tensor<7xf32>
}
func.func @next(%arg0: tensor<3xf32>, %arg1: tensor<7xf32>) -> tensor<6xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<6xf32>
  return %0 : tensor<6xf32>
}
func.func @finalize(%arg0: tensor<6xf32>, %arg1: tensor<2xf32>) -> tensor<5xf32> {
  %0 = builtin.unrealized_conversion_cast to tensor<5xf32>
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: testGeneratorDatasetRegionWithComplexBlocks
func.func @testGeneratorDatasetRegionWithComplexBlocks(%arg0: tensor<4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<!tf_type.resource>, %arg3: tensor<2xf32>) {
  // CHECK: "tf.GeneratorDataset"
  // CHECK-NOT: @init
  // CHECK-NOT: @next
  // CHECK-NOT: @finalize
  // CHECK: -> tensor<!tf_type.variant>
  // CHECK: return
  %0 = "tf.GeneratorDatasetRegion"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<4xf32>):
    %sum = "tf.Add"(%arg4, %arg4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %1 = func.call @init(%sum) : (tensor<4xf32>) -> tensor<7xf32>
    "tf.Yield"(%1) : (tensor<7xf32>) -> ()
  }, {
  ^bb0(%arg4: tensor<7xf32>, %arg5: tensor<3xf32>):
    %1 = func.call @next(%arg5, %arg4) : (tensor<3xf32>, tensor<7xf32>) -> tensor<6xf32>
    "tf.Yield"(%1) : (tensor<6xf32>) -> ()
  }, {
  ^bb0(%arg4: tensor<6xf32>, %arg5: tensor<2xf32>):
    %1 = func.call @finalize(%arg4, %arg5) : (tensor<6xf32>, tensor<2xf32>) -> tensor<5xf32>
    %sum = "tf.Add"(%1, %1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    "tf.Yield"(%sum) : (tensor<5xf32>) -> ()
  }) {device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0", metadata = "", operandSegmentSizes = array<i32: 1, 2, 1>, output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string]} : (tensor<4xf32>, tensor<3xf32>, tensor<!tf_type.resource>, tensor<2xf32>) -> tensor<!tf_type.variant>
  return
}

// -----

func.func private @tf.WhileRegion_body(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  %1 = builtin.unrealized_conversion_cast to tensor<i32>
  %2 = builtin.unrealized_conversion_cast to tensor<f32>
  return %1, %2 : tensor<i32>, tensor<f32>
}
func.func private @tf.WhileRegion_cond(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = builtin.unrealized_conversion_cast to tensor<i1>
  return %0 : tensor<i1>
}
// CHECK-LABEL: testNameCollision
func.func @testNameCollision(%arg0: tensor<i32>) {
  %1 = builtin.unrealized_conversion_cast to tensor<i32>
  %2 = builtin.unrealized_conversion_cast to tensor<f32>
  // CHECK: "tf.While"
  // CHECK-SAME: body = @tf.WhileRegion_body_1
  // CHECK-SAME: cond = @tf.WhileRegion_cond_0
  %3:2 = "tf.WhileRegion"(%1, %2) <{is_stateless = false}> ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<f32>):
    %8 = func.call @tf.WhileRegion_cond(%arg1) : (tensor<i32>) -> tensor<i1>
    "tf.Yield"(%8, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<f32>) -> ()
  }, {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<f32>):
    %8:2 = func.call @tf.WhileRegion_body(%arg1, %arg2) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    "tf.Yield"(%8#0, %8#1) : (tensor<i32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  return
}

// -----

func.func private @my_cond(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = builtin.unrealized_conversion_cast to tensor<i1>
  return %0 : tensor<i1>
}
func.func private @my_body(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  return %arg0, %arg1 : tensor<i32>, tensor<f32>
}
// CHECK-LABEL: testConditionWithPassthroughArgs
func.func @testConditionWithPassthroughArgs(%arg1: tensor<i32>, %arg2: tensor<f32>) {
  // CHECK: "tf.While"
  // CHECK-SAME: body = @my_body
  // CHECK-SAME: cond = @my_cond
  %3:2 = "tf.WhileRegion"(%arg1, %arg2) <{is_stateless = false}> ({
  ^bb0(%barg1: tensor<i32>, %barg2: tensor<f32>):
    %8 = func.call @my_cond(%barg1, %barg2) : (tensor<i32>, tensor<f32>) -> tensor<i1>
    "tf.Yield"(%8, %barg1, %barg2) : (tensor<i1>, tensor<i32>, tensor<f32>) -> ()
  }, {
  ^bb0(%barg1: tensor<i32>, %barg2: tensor<f32>):
    %r1, %r2 = func.call @my_body(%barg1, %barg2) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
    "tf.Yield"(%r1, %r2) : (tensor<i32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  return
}
