// RUN: sdy_opt %s -xla-sdy-open-while-free-vars-sharding 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["a"=2]>
sdy.mesh @mesh2 = <["b"=2]>

// CHECK-LABEL: func @while_with_free_variables
func.func @while_with_free_variables(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a"}, {}]>},
    %arg2: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[C0:.*]] = mhlo.constant dense<0>
  // CHECK-NEXT: %[[C1:.*]] = mhlo.constant dense<1>
  // CHECK-NEXT: %[[C32:.*]] = mhlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32>
  // CHECK-NEXT: %[[ADD_0:.*]] = mhlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{}, {"b"}]>]>}
  // CHECK-NEXT: %[[SC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh2, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   mhlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = mhlo.add %iterArg_0, %[[C1]]
  // CHECK-NEXT:   %[[ADD_2:.*]] = mhlo.add %iterArg, %[[SC_0]]
  // CHECK-NEXT:   %[[ADD_3:.*]] = mhlo.add %[[ADD_2]], %arg2
  // CHECK-NEXT:   %[[ADD_4:.*]] = mhlo.add %[[ADD_3]], %[[SC_1]]
  // CHECK-NEXT:   mhlo.return %[[ADD_4]], %[[ADD_1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[ADD_0]], %[[WHILE]]#0
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  %2 = mhlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32> : tensor<i32>
  %3 = mhlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{}, {"b"}]>]>} : tensor<32x96xf32>
  %4:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %5 = mhlo.compare LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %5 : tensor<i1>
  } do {
    %5 = mhlo.add %iterArg_0, %1 : tensor<i32>
    %6 = mhlo.add %iterArg, %arg1 : tensor<32x96xf32>
    %7 = mhlo.add %6, %arg2 : tensor<32x96xf32>
    %8 = mhlo.add %7, %3 : tensor<32x96xf32>
    mhlo.return %8, %5 : tensor<32x96xf32>, tensor<i32>
  }
  return %3, %4#0 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @free_var_used_in_multiple_while_ops
func.func @free_var_used_in_multiple_while_ops(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a"}, {}]>})
    -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = mhlo.constant dense<0>
  // CHECK-NEXT: %[[C32:.*]] = mhlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32>
  // CHECK-NEXT: %[[SC_0:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE_0:.*]]:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   mhlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_0:.*]] = mhlo.add %iterArg, %[[SC_0]]
  // CHECK-NEXT:   mhlo.return %[[ADD_0]], %iterArg_0
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg1 <@mesh1, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE_1:.*]]:2 = mhlo.while(%iterArg = %[[WHILE_0]]#0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   mhlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_1:.*]] = mhlo.add %iterArg, %[[SC_1]]
  // CHECK-NEXT:   mhlo.return %[[ADD_1]], %iterArg_0
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE_1]]#0
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, []>]>} dense<32> : tensor<i32>
  %2:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = mhlo.compare LT, %iterArg_0, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %4 : tensor<i1>
  } do {
    %4 = mhlo.add %iterArg, %arg1 : tensor<32x96xf32>
    mhlo.return %4, %iterArg_0 : tensor<32x96xf32>, tensor<i32>
  }
  %3:2 = mhlo.while(%iterArg = %2#0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = mhlo.compare LT, %iterArg_0, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    mhlo.return %4 : tensor<i1>
  } do {
    %4 = mhlo.add %iterArg, %arg1 : tensor<32x96xf32>
    mhlo.return %4, %iterArg_0 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}
