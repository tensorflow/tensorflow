// RUN: tf-opt -loop-invariant-code-motion %s -o - | FileCheck %s

// CHECK: while_1([[ARG0:%[^ :]*]]: tensor<i32>, [[ARG1:%[^ :]*]]: tensor<1xf32>)
func @while_1(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: [[CST:%[^ ]*]] = arith.constant dense<1> : tensor<i32>
  // CHECK: "tfl.while"([[ARG0]], [[ARG1]])
  // CHECK: (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  %0:2 = "tfl.while"(%arg0, %arg1) (
    // cond
    {
    ^bb0(%condArg0: tensor<*xi32>, %condArg1: tensor<*xf32>):
      %0 = "arith.constant" () {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tfl.greater"(%condArg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
    },
    // body
    {
    ^bb0(%bodyArg0: tensor<*xi32>, %bodyArg1: tensor<*xf32>):
      %0 = "arith.constant" () {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tfl.sub"(%bodyArg0, %0) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %bodyArg1, %bodyArg1 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
    }
  ) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  return %0#1 : tensor<1xf32>
}
