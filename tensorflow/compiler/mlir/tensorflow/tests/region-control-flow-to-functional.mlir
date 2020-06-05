// RUN: tf-opt %s -tf-region-control-flow-to-functional -split-input-file
//| FileCheck %s --dump-input=fail

// CHECK: func @tf.IfRegion_else(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
// CHECK-NEXT:   "tf.Neg"
// CHECK: func @tf.IfRegion_then(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
// CHECK-NEXT:   "tf.Abs"
func @testSimple(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.If"{{.+}}else_branch = @tf.IfRegion_else{{.+}}then_branch = @tf.IfRegion_then
  %0 = "tf.IfRegion"(%arg0) ({
    %1 = "tf.Abs"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1) : (tensor<*xf32>) -> ()
    }, {
    %2 = "tf.Neg"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%2) : (tensor<*xf32>) -> ()
    }) { is_stateless = true } :  (tensor<i1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// Use if condition inside the regions
// CHECK: func @tf.IfRegion_else(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32> attributes {sym_visibility = "private"}
// CHECK-NEXT: "tf.Select"(%arg0, %arg2, %arg3)
// CHECK: func @tf.IfRegion_then(%arg0: tensor<i1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> tensor<2xf32> attributes {sym_visibility = "private"}
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

// Constant sinking

// CHECK: func @tf.IfRegion_else() -> tensor<2xf32>
// CHECK-NEXT: constant dense<1.0
// CHECK: func @tf.IfRegion_then() -> tensor<2xf32>
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
// CHECK: func @tf.IfRegion1_else
// CHECK-NEXT: "tf.Acos"
// CHECK-NEXT: "tf.Abs"

// CHECK: func @tf.IfRegion1_then
// CHECK-NEXT: "tf.LogicalNot"
// CHECK-NEXT: "tf.Asin"
// CHECK-NEXT: "tf.If"({{.+}}) {else_branch = @tf.IfRegion_else, {{.+}} then_branch = @tf.IfRegion_then}

// CHECK: func @tf.IfRegion_else
// CHECK-NEXT: "tf.Neg"
// CHECK: func @tf.IfRegion_then
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

// Match existing function->Region pattern (simple)
func @testIf1Then(tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
func @testIf1Else(tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
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

// Match existing function->Region pattern (with casts)

func @testIf1Then(tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
func @testIf1Else(tensor<*xf32>) -> tensor<*xf32> attributes {sym_visibility = "private"}
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

