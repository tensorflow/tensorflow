// RUN: tf-opt %s -tf-functional-control-flow-to-regions -split-input-file | FileCheck %s --dump-input=fail

// CHECK: func @testIf1Then{{.+}}
// CHECK: func @testIf1Else{{.+}}
func @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>)
func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: [[Result0:%.*]] = call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: [[Result1:%.*]] = call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  return %0 : tensor<*xf32>
}

// -----

// With mismatching input types

// CHECK: func @testIf1Then{{.+}}
// CHECK: func @testIf1Else{{.+}}
func @testIf1Then(tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>)
func @testIf2Result(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: "tf.Cast"
  // CHECK: [[Result0:%.*]] = call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: "tf.Cast"
  // CHECK: [[Result1:%.*]] = call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  return %0 : tensor<2xf32>
}

// -----

// No inputs, some outputs
// CHECK: func @testIf1Then{{.+}}
// CHECK: func @testIf1Else{{.+}}
func @testIf1Then() -> tensor<*xf32>
func @testIf1Else() -> tensor<*xf32>

// CHECK-LABEL: func @testIfNoInputs(%arg0: tensor<i1>)
func @testIfNoInputs(%arg0: tensor<i1>) -> tensor<2xf32> {
  %0 = "tf.If"(%arg0) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>) -> tensor<2xf32>

  // CHECK: "tf.IfRegion"
  // CHECK: [[Result0:%.*]] = call @testIf1Then
  // CHECK: "tf.Yield"([[Result0]])
  // CHECK: [[Result1:%.*]] = call @testIf1Else
  // CHECK: "tf.Yield"([[Result1]])
  return %0 : tensor<2xf32>
}

// -----

// No outputs, some inputs
// CHECK: func @testIf1Then{{.+}}
// CHECK: func @testIf1Else{{.+}}
func @testIf1Then(tensor<*xf32>) -> ()
func @testIf1Else(tensor<*xf32>) -> ()

// CHECK-LABEL: func @testIfNoResult(%arg0: tensor<i1>, %arg1: tensor<2xf32>)
func @testIfNoResult(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> () {
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
  return
}

// -----
// No outputs, No inputs
// CHECK: func @testIf1Then{{.+}}
// CHECK: func @testIf1Else{{.+}}
func @testIf1Then() -> ()
func @testIf1Else() -> ()

// CHECK-LABEL: func @testIfNoInputAndNoResult(%arg0: tensor<i1>)
func @testIfNoInputAndNoResult(%arg0: tensor<i1>) -> () {
  "tf.If"(%arg0) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>) -> ()

  // CHECK: "tf.IfRegion"
  // CHECK: call @testIf1Then
  // CHECK: "tf.Yield"()
  // CHECK: call @testIf1Else
  // CHECK: "tf.Yield"()
  return
}

