// RUN: tf-opt -split-input-file -loop-invariant-code-motion %s | FileCheck %s

// CHECK: while_1([[ARG0:%[^ :]*]]: tensor<i32>, [[ARG1:%[^ :]*]]: tensor<1xf32>)
func @while_1(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: [[CST:%[^ ]*]] = constant dense<1> : tensor<i32>
  // CHECK: "tf.WhileRegion"([[ARG0]], [[ARG1]])
  // CHECK: (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    // cond
    {
    ^bb0(%condArg0: tensor<*xi32>, %condArg1: tensor<*xf32>):
      %0 = "std.constant" () {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tf.NotEqual"(%condArg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%1) : (tensor<i1>) -> ()
    },
    // body
    {
    ^bb0(%bodyArg0: tensor<*xi32>, %bodyArg1: tensor<*xf32>):
      %0 = "std.constant" () {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tf.Sub"(%bodyArg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = "tf.Add"(%bodyArg1, %bodyArg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      "tf.Yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
    }
  ) {is_stateless = false} : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  return %0#1 : tensor<1xf32>
}

// -----

// Test WhileRegionOp::isDefinedOutsideOfLoop
// CHECK-LABEL: testWhileRegionisDefinedOutsideOfLoop
func @testWhileRegionisDefinedOutsideOfLoop(%arg0 : tensor<4xf32>, %arg1 : tensor<i32>) -> tensor<4xf32> {
  %a = "tf.Neg"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %b = "tf.Abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // Verify that the Div and Mul are hoisted out of the body
  // CHECK: "tf.Div"
  // CHECK: constant dense<2.200000e+01>
  // CHECK: "tf.Mul"
  // CHECK: "tf.WhileRegion"
  // Verify that Add and Sub is not hoisted out
  // CHECK: "tf.Add"
  // CHECK: "tf.Sub"
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

      // Some loop invariant math
      %li0 = "tf.Div"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %cst = constant dense<22.0> : tensor<f32>
      %li1 = "tf.Mul"(%li0, %cst) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

      %final = "tf.Add"(%add, %li1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "tf.Yield"(%final, %sub) : (tensor<4xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)

  return %0#0 : tensor<4xf32>
}
