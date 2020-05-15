// RUN: xla-opt %s -xla-hlo-sink-constants-to-control-flow | FileCheck %s --dump-input=fail

// Tests sinking constants to a while loop.

// CHECK-LABEL: func @sink_const_to_while
func @sink_const_to_while(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK-NEXT: xla_hlo.while
  %c0 = xla_hlo.constant dense<1> : tensor<i64>
  %c1 = xla_hlo.constant dense<2> : tensor<i64>
  %0 = "xla_hlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    // CHECK: %[[ARG1A:.+]]: tensor<i64>
    // CHECK: %[[C0:.+]] = xla_hlo.constant dense<1> : tensor<i64>
    // CHECK: "xla_hlo.compare"(%[[C0]], %[[ARG1A]])
    %1 = "xla_hlo.compare"(%c0, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "xla_hlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i64>):
    // CHECK: %[[ARG1B:.+]]: tensor<i64>
    // CHECK-DAG: %[[C1:.+]] = xla_hlo.constant dense<2> : tensor<i64>
    // CHECK-DAG: %[[ADD0:.+]] = xla_hlo.add %[[ARG1B]], %[[ARG1B]]
    %2 = xla_hlo.add %arg1, %arg1 : tensor<i64>
    // CHECK: %[[ADD1:.+]] = xla_hlo.add %[[C1]], %[[ADD0]]
    %3 = xla_hlo.add %c1, %2 : tensor<i64>
    // CHECK: %[[ADD2:.+]] = xla_hlo.add %[[C1]], %[[ADD1]]
    %4 = xla_hlo.add %c1, %3 : tensor<i64>
    "xla_hlo.return"(%4) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// Tests sinking constants to a conditional op.

// CHECK-LABEL: func @sink_const_to_conditional
func @sink_const_to_conditional(%arg0: tensor<i64>) -> tensor<i64> {
  %c0 = xla_hlo.constant dense<1> : tensor<i64>
  %c1 = xla_hlo.constant dense<2> : tensor<i64>
  %0 = "xla_hlo.compare"(%arg0, %c0) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %1 = "xla_hlo.tuple"(%arg0) : (tensor<i64>) -> tuple<tensor<i64>>
  // CHECK: xla_hlo.conditional
  %2 = "xla_hlo.conditional"(%0, %1, %1) ( {
  ^bb0(%arg1: tuple<tensor<i64>>):
    // CHECK: %[[C0:.+]] = xla_hlo.constant dense<1> : tensor<i64>
    %3 = "xla_hlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    // CHECK: %[[ADD0:.+]] = xla_hlo.add %[[C0]],
    %4 = xla_hlo.add %c0, %3 : tensor<i64>
    %5 = "xla_hlo.tuple"(%4) : (tensor<i64>) -> tuple<tensor<i64>>
    "xla_hlo.return"(%5) : (tuple<tensor<i64>>) -> ()
  },  {
  ^bb0(%arg1: tuple<tensor<i64>>):
    // CHECK: %[[C1:.+]] = xla_hlo.constant dense<2> : tensor<i64>
    %6 = "xla_hlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
    // CHECK: %[[ADD1:.+]] = xla_hlo.add %[[C1]],
    %7 = xla_hlo.add %c1, %6 : tensor<i64>
    %8 = "xla_hlo.tuple"(%7) : (tensor<i64>) -> tuple<tensor<i64>>
    "xla_hlo.return"(%8) : (tuple<tensor<i64>>) -> ()
  }) : (tensor<i1>, tuple<tensor<i64>>, tuple<tensor<i64>>) -> tuple<tensor<i64>>
  %9 = "xla_hlo.get_tuple_element"(%2) {index = 0 : i32} : (tuple<tensor<i64>>) -> tensor<i64>
  return %9 : tensor<i64>
}
