// Test to verify loop outlining.

// RUN: tf-opt --tfl-while-loop-outline %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @while
func @while() -> tensor<1xf32>
    attributes {tf.entry_function = {outputs = "result"}} {
  %cst = constant dense<1> : tensor<i32> loc("dec")
  %arg0 = constant dense<5> : tensor<i32> loc("N")
  %arg1 = constant dense<3.0> : tensor<1xf32> loc("val")
  %0:2 = "tfl.while"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_cond
      %cst_0 = constant dense<0> : tensor<i32>
      %1 = "tfl.greater"(%arg2, %cst_0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_body
      %1 = "tfl.sub"(%arg2, %cst) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %arg3, %arg3 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  return %0#1 : tensor<1xf32>
}
// CHECK-LABEL: func @WhileOp_cond(
// CHECK: tfl.greater
// CHECK-LABEL: func @WhileOp_body(
// CHECK: tfl.sub
// CHECK: tfl.add
