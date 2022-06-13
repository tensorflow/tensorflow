// RUN: mlir-hlo-opt %s -split-input-file -canonicalize | FileCheck %s


// CHECK-LABEL: func @loop_invariants
module  {
  func.func @loop_invariants(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
    // The first operand is used directly as an implicit capture in the return
    // of the body, the third operand is loop carried: they both can be
    // eliminated, ony the second operand is really a loop-carried value.
    // CHECK: %[[WHILE:.*]] = mhlo.while
    // CHECK-SAME: (%[[ITER_ARG:.*]] = %arg2)
    %0:3 = "mhlo.while"(%arg0, %arg2, %arg3) ({
    ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
      // CHECK: mhlo.compare
      // CHECK-SAME: %[[ITER_ARG]], %arg3
      %1 = "mhlo.compare"(%arg5, %arg6) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
      // CHECK: %[[ADD:.*]] = mhlo.add %[[ITER_ARG]], %arg0
      %1 = mhlo.add %arg5, %arg4 : tensor<i32>
     // This op is dead, its removal will enable the canonicalization of the while op.
      %2 = "mhlo.tuple"(%arg4, %1, %arg6) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>, tensor<i32>>
      // CHECK: mhlo.return
      // CHECK-SAME: %[[ADD]]
      "mhlo.return"(%arg0, %1, %arg6) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    // CHECK: return %arg0, %[[WHILE]], %arg3
    func.return %0#0, %0#1, %0#2 : tensor<i32>, tensor<i32>, tensor<i32>
  }
}

// -----

// CHECK-LABEL: func @dead_loop
module  {
  func.func @dead_loop(%arg0: tensor<i32>) -> tensor<i32> {
    // The following loop will always return its operand which is carried over
    // from one iteration to the next as-is, that is: we assume that loops
    // always terminate.
    // CHECK-NOT: mhlo.while
    %0 = "mhlo.while"(%arg0) ({
    ^bb0(%arg1: tensor<i32>):
      %1 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<"comparison_direction LT">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i32>):
      "mhlo.return"(%arg1) : (tensor<i32>) -> ()
    }) : (tensor<i32>) -> (tensor<i32>)
    func.return %0 : tensor<i32>
  }
}

// -----

// CHECK-LABEL: func @fold_constant_cond
func.func @fold_constant_cond(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
// CHECK-NOT: while
// CHECK: return %arg0, %arg
  %0:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %arg1) : tensor<4xf32>, tensor<4xf32>
   cond {
    %cst = arith.constant dense<false> : tensor<i1>
    "mhlo.return"(%cst) : (tensor<i1>) -> ()
  } do {
    %1 = mhlo.add %iterArg, %iterArg_0 : tensor<4xf32>
    "mhlo.return"(%1, %1) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}