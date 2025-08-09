// RUN: sdy_opt %s -xla-sdy-open-while-free-vars-sharding 2>&1 | FileCheck %s

// Verify calling this pass a second time is a no-op.
// RUN: sdy_opt %s -xla-sdy-open-while-free-vars-sharding -xla-sdy-open-while-free-vars-sharding 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["a"=2]>
sdy.mesh @mesh2 = <["b"=2]>

// CHECK-LABEL: func @while_with_free_variables
func.func @while_with_free_variables(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a"}, {}]>},
    %arg2: tensor<32x96xf32>,
    %arg3: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{?}, {?}]>})
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1>
  // CHECK-NEXT: %[[C32:.*]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{}, {"b"}]>]>}
  // CHECK-NEXT: %[[SC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh2, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_2, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg_2, %[[C1]]
  // CHECK-NEXT:   %[[ADD_2:.*]] = stablehlo.add %iterArg, %[[SC_0]]
  // CHECK-NEXT:   %[[ADD_3:.*]] = stablehlo.add %[[ADD_2]], %arg2
  // CHECK-NEXT:   %[[ADD_4:.*]] = stablehlo.add %[[ADD_3]], %[[SC_1]]
  // CHECK-NEXT:   %[[ADD_5:.*]] = stablehlo.add %[[ADD_4]], %arg3
  // CHECK-NEXT:   stablehlo.return %[[ADD_5]], %[[ADD_1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[ADD_0]], %[[WHILE]]#0
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32> : tensor<i32>
  %3 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{}, {"b"}]>]>} : tensor<32x96xf32>
  %4:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %5 = stablehlo.compare LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %5 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %6 = stablehlo.add %iterArg, %arg1 : tensor<32x96xf32>
    %7 = stablehlo.add %6, %arg2 : tensor<32x96xf32>
    %8 = stablehlo.add %7, %3 : tensor<32x96xf32>
    %9 = stablehlo.add %8, %arg3 : tensor<32x96xf32>
    stablehlo.return %9, %5 : tensor<32x96xf32>, tensor<i32>
  }
  return %3, %4#0 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @free_var_used_in_multiple_while_ops
func.func @free_var_used_in_multiple_while_ops(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a"}, {}]>})
    -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK-NEXT: %[[C32:.*]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32>
  // CHECK-NEXT: %[[SC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE_0:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_1, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_0:.*]] = stablehlo.add %iterArg, %[[SC_0]]
  // CHECK-NEXT:   stablehlo.return %[[ADD_0]], %iterArg_1
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE_1:.*]]:2 = stablehlo.while(%iterArg = %[[WHILE_0]]#0, %iterArg_1 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_1, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg, %[[SC_1]]
  // CHECK-NEXT:   stablehlo.return %[[ADD_1]], %iterArg_1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE_1]]#0
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32> : tensor<i32>
  %2:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg, %arg1 : tensor<32x96xf32>
    stablehlo.return %4, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  %3:2 = stablehlo.while(%iterArg = %2#0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg, %arg1 : tensor<32x96xf32>
    stablehlo.return %4, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}
