// RUN: litert-opt -tfl-reduce-while %s | FileCheck %s

// The original func we want to optimize is:
//
// func increase_3rd_operand_3_times():
//   S = (1, 0, 0)
//   whlie (S[2] < 3) {
//     s0 = S[0] * 2
//     s1 = S[0] + S[1]
//     s2 = S[2] + 1
//     S = (s0, s1, s2)
//   }
//   return S[2]
// }
//
// Since only S[2] is returned and the computation of final S[2] does not depend
// on S[0] and S[1]. The func can be optimized to
//
// func increase_3rd_operand_3_times():
//   s2 = 0
//   whlie (s2 < 3) {
//     s2 = s2 + 1
//   }
//   return s2
// }
//
// CHECK-LABEL:   func @increase_3rd_operand_3_times() -> tensor<i32> {
// CHECK-DAG:       %[[CST_0:.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       %[[CST_2:.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:       %[[VAL_0:.*]] = "tfl.while"(%[[CST_2]]) ({
// CHECK-DAG:       ^bb0(%[[A2_COND:.*]]: tensor<i32>):
// CHECK-DAG:         %[[CST_3:.*]] = arith.constant dense<3> : tensor<i32>
// CHECK-DAG:         %[[VAL_1:.*]] = tfl.less(%[[A2_COND]], %[[CST_3]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-DAG:         "tfl.yield"(%[[VAL_1]]) : (tensor<i1>) -> ()
// CHECK-DAG:       },  {
// CHECK-DAG:       ^bb0(%[[A2_BODY:.*]]: tensor<i32>):
// CHECK-DAG:         %[[CST_4:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK-DAG:         %[[VAL_2:.*]] = tfl.add %[[A2_BODY]], %[[CST_4]] {fused_activation_function = "NONE"} : tensor<i32>
// CHECK-DAG:         "tfl.yield"(%[[VAL_2]]) : (tensor<i32>) -> ()
// CHECK-DAG:       }) : (tensor<i32>) -> tensor<i32>
// CHECK-DAG:       return %[[VAL_0]] : tensor<i32>
// CHECK-DAG:     }
func.func @increase_3rd_operand_3_times() -> tensor<i32> {
  %cst_0 = "arith.constant" () {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "arith.constant" () {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %cst_2 = "arith.constant" () {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %0:3 = "tfl.while"(%cst_0, %cst_1, %cst_2) (
    // cond
    {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>):
      %cst_3 = "arith.constant" () {value = dense<3> : tensor<i32>} : () -> tensor<i32>
      %1 = "tfl.less"(%arg2, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
    },
    // body
    {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>):
      %2 = "tfl.add"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %3 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %cst_4 = "arith.constant" () {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %4 = "tfl.add"(%arg2, %cst_4) {fused_activation_function = "NONE"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tfl.yield"(%2, %3, %4) : (tensor<f32>, tensor<f32>, tensor<i32>) -> ()
    }
  ) : (tensor<f32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<f32>, tensor<i32>)
  func.return %0#2 : tensor<i32>
}
