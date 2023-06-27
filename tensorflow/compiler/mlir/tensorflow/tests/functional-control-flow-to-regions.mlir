// RUN: tf-opt %s -tf-functional-control-flow-to-regions -split-input-file | FileCheck %s

// Simple If
// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>)
func.func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false,
    _attr0 = 10, _attr1 = true, attr2 = "hello"
  } : (tensor<i1>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: [[Result0:%.*]] = func.call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: [[Result1:%.*]] = func.call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  // CHECK: _attr0 = 10
  // CHECK-SAME: _attr1 = true
  // CHECK-SAME: _else_func_name = "testIf1Else"
  // CHECK-SAME: _then_func_name = "testIf1Then"
  // CHECK-NOT: attr2 =
  // CHECK-NOT: else_branch
  // CHECK-SAME: is_stateless = false
  // CHECK-NOT: then_branch
  // CHECK-SAME: }
  func.return %0 : tensor<*xf32>
}

// -----

// If with mismatching input types

// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>)
func.func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: "tf.Cast"
  // CHECK: [[Result0:%.*]] = func.call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: "tf.Cast"
  // CHECK: [[Result1:%.*]] = func.call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  func.return %0 : tensor<2xf32>
}

// -----

// If with no inputs, some outputs
// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then() -> tensor<*xf32>
func.func private @testIf1Else() -> tensor<*xf32>

// CHECK-LABEL: func @testIfNoInputs(%arg0: tensor<i1>)
func.func @testIfNoInputs(%arg0: tensor<i1>) -> tensor<2xf32> {
  %0 = "tf.If"(%arg0) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>) -> tensor<2xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: [[Result0:%.*]] = func.call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: [[Result1:%.*]] = func.call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  func.return %0 : tensor<2xf32>
}

// -----

// If with no outputs, some inputs
// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then(tensor<*xf32>) -> ()
func.func private @testIf1Else(tensor<*xf32>) -> ()

// CHECK-LABEL: func @testIfNoResult(%arg0: tensor<i1>, %arg1: tensor<2xf32>)
func.func @testIfNoResult(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> () {
  "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> ()

  // CHECK: "tf.IfRegion"
  // CHECK: "tf.Cast"
  // CHECK: call @testIf1Then
  // CHECK: "tf.Yield"()
  // CHECK: "tf.Cast"
  // CHECK: call @testIf1Else
  // CHECK: "tf.Yield"()
  func.return
}

// -----

// If with no outputs, No inputs
// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then() -> ()
func.func private @testIf1Else() -> ()

// CHECK-LABEL: func @testIfNoInputAndNoResult(%arg0: tensor<i1>)
func.func @testIfNoInputAndNoResult(%arg0: tensor<i1>) -> () {
  "tf.If"(%arg0) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>) -> ()

  // CHECK: "tf.IfRegion"
  // CHECK: call @testIf1Then
  // CHECK: "tf.Yield"()
  // CHECK: call @testIf1Else
  // CHECK: "tf.Yield"()
  func.return
}

// -----

// If with non tensor<i1> condition

// Simple If
// CHECK: func private @testIf1Then{{.+}}
// CHECK: func private @testIf1Else{{.+}}
func.func private @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIf1Else(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i32>, %arg1: tensor<*xf32>)
func.func @testIf1Result(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: [[ToBool:%.*]] = "tf.ToBool"
  // CHECK: "tf.IfRegion"([[ToBool]])
  func.return %0 : tensor<*xf32>
}

// -----

func.func private @branch_0(%arg0: tensor<!tf_type.resource>) -> tensor<*xf32>
func.func private @branch_1(%arg0: tensor<!tf_type.resource>) -> tensor<*xf32>

// CHECK-LABEL: func @testCase(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<1x2x3xf32>>>)
func.func @testCase(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32> {
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch_0, @branch_1], is_stateless = false} : (tensor<i32>, tensor<!tf_type.resource<tensor<1x2x3xf32>>>) -> tensor<1x2x3xf32>

  // CHECK: [[Result0:%.*]] = "tf.CaseRegion"
  // CHECK: [[Result1:%.*]] = func.call @branch_0
  // CHECK: "tf.Yield"([[Result1]])
  // CHECK: [[Result2:%.*]] = func.call @branch_1
  // CHECK: "tf.Yield"([[Result2]])
  // CHECK: return [[Result0]]
  func.return %0 : tensor<1x2x3xf32>
}

// -----

// Simple While
func.func private @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = true,
    _attr0 = 10, _attr1 = true, attr2 = "hello"
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  // CHECK: [[Result0:%.*]] = "tf.WhileRegion"
  // CHECK: [[Result1:%.*]] = func.call @testWhileCond
  // CHECK: "tf.Yield"([[Result1]])
  // CHECK: [[Result2:%.*]] = func.call @testWhileBody
  // CHECK: "tf.Yield"([[Result2]])
  // CHECK: _attr0 = 10
  // CHECK-SAME: _attr1 = true
  // CHECK-NOT: attr2 =
  // CHECK-NOT: cond =
  // CHECK-NOT: body =
  // CHECK-SAME: is_stateless = true
  // CHECK: return [[Result0]]
  func.return %1 : tensor<*xf32>
}

// -----

// While with no inputs & outputs
func.func private @testWhileCond() -> (tensor<i1>)
func.func private @testWhileBody() -> ()

// CHECK-LABEL: func @testWhileResultNoIO
func.func @testWhileResultNoIO() -> () {
  "tf.While"() {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : () -> ()

  // CHECK: "tf.WhileRegion"
  // CHECK: [[Result1:%.*]] = func.call @testWhileCond
  // CHECK: "tf.Yield"([[Result1]])
  // CHECK: call @testWhileBody
  // CHECK: "tf.Yield"()
  func.return
}

// -----

// While with type mismatch
func.func private @testWhileCond(tensor<4xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<4xf32>) -> (tensor<4xf32>)

// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  // CHECK: [[Result0:%.*]] = "tf.WhileRegion"
  // CHECK: ^bb0(%[[CARG0:.*]]: tensor<4xf32>
  // CHECK: [[Result1:%.*]] = func.call @testWhileCond(%[[CARG0]])
  // CHECK: "tf.Yield"([[Result1]])
  // CHECK: ^bb0(%[[BARG0:.*]]: tensor<4xf32>
  // CHECK: [[Result2:%.*]] = func.call @testWhileBody(%[[BARG0]])
  // CHECK: "tf.Yield"([[Result2]])
  // CHECK: return [[Result0]]
  func.return %1 : tensor<*xf32>
}

// -----

// While with non tensor<i1> condition
func.func private @testWhileCond(tensor<*xf32>) -> (tensor<f32>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = true,
    _attr0 = 10, _attr1 = true, attr2 = "hello"
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  // CHECK: [[Result0:%.*]] = "tf.WhileRegion"
  // CHECK: [[Result1:%.*]] = func.call @testWhileCond
  // CHECK: [[ToBool:%.*]] = "tf.ToBool"([[Result1]])
  // CHECK: "tf.Yield"([[ToBool]])
  // CHECK: [[Result2:%.*]] = func.call @testWhileBody
  // CHECK: "tf.Yield"([[Result2]])
  // CHECK: return [[Result0]]
  func.return %1 : tensor<*xf32>
}

// -----

func.func private @then_branch() -> ()
func.func private @else_branch() -> ()

// Test tf.If device is preserved.
// CHECK-LABEL: func @testIfDevice
func.func @testIfDevice(%arg0: tensor<i1>) {
  "tf.If"(%arg0) {then_branch = @then_branch, else_branch = @else_branch, is_stateless = false, device = "/device:CPU:0"} : (tensor<i1>) -> ()

  // CHECK: "tf.IfRegion"
  // CHECK: device = "/device:CPU:0"
  func.return
}

// -----

func.func private @cond() -> tensor<i1>
func.func private @body() -> ()

// Test tf.While device is preserved.
// CHECK-LABEL: func @testWhileDevice
func.func @testWhileDevice() {
  "tf.While"() {cond = @cond, body = @body, is_stateless = false, device = "/device:CPU:0"} : () -> ()

  // CHECK: "tf.WhileRegion"
  // CHECK: device = "/device:CPU:0"
  func.return
}
