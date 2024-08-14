// RUN: sdy_opt %s -xla-sdy-round-trip-import-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: module @multiple_func_result_shardings
module @multiple_func_result_shardings attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = \22#sdy.mesh<\\22a\\22=8, \\22b\\22=8, \\22c\\22=8>\22}"}} {
  // CHECK: sdy.mesh @mesh = <"a"=8, "b"=8, "c"=8>

  // CHECK-LABEL: func @func_results_with_sharding
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME: %arg1: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p1]>},
  // CHECK-SAME: %arg2: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}p0]>}
  // CHECK-SAME: ) -> (
  // CHECK-SAME: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0]>},
  // CHECK-SAME: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p1]>},
  // CHECK-SAME: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}p0]>},
  // CHECK-SAME: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p3]>}) {
  // CHECK-NEXT:   return %arg0, %arg1, %arg0, %arg1, %arg2 : tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>
  // CHECK-NEXT: }
  func.func @func_results_with_sharding(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22b\22}p2]>"}},
    %arg1: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22a\22}p1]>"}},
    %arg2: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22c\22}p0]>"}}
  ) -> (tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>) {
    %0 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p0]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22b\22}p2]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %2 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p1]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22c\22}p0]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %4 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg2) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p3]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %0, %1, %2, %3, %4 : tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>
  }

  // CHECK-LABEL: func @while_with_free_variables
  func.func @while_with_free_variables(
      %arg0: tensor<32x96xf32>,
      %arg1: tensor<32x96xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}})
      -> tensor<32x96xf32> {
    // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
    // CHECK-NEXT: %[[C1:.*]] = sdy.constant dense<1>
    // CHECK-NEXT: %[[C32:.*]] = sdy.constant dense<32>
    // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]>
    // CHECK-NEXT: %[[WHILE:.*]]:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
    // CHECK-NEXT:   cond {
    // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
    // CHECK-NEXT:   mhlo.return %[[COND]]
    // CHECK-NEXT: } do {
    // CHECK-NEXT:   %[[ADD_0:.*]] = mhlo.add %iterArg_0, %[[C1]]
    // CHECK-NEXT:   %[[ADD_1:.*]] = mhlo.add %iterArg, %[[SC]]
    // CHECK-NEXT:   mhlo.return %[[ADD_1]], %[[ADD_0]]
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[WHILE]]#0
    %0 = mhlo.constant dense<0> : tensor<i32>
    %1 = mhlo.constant dense<1> : tensor<i32>
    %2 = mhlo.constant dense<32> : tensor<i32>
    %3:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
      cond {
      %4 = mhlo.compare LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      mhlo.return %4 : tensor<i1>
    } do {
      %4 = mhlo.add %iterArg_0, %1 : tensor<i32>
      %5 = mhlo.add %iterArg, %arg1 : tensor<32x96xf32>
      mhlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
    }
    return %3#0 : tensor<32x96xf32>
  }

  // CHECK-LABEL: func @while_with_sinked_constants
  func.func @while_with_sinked_constants(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
    // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
    // CHECK-NEXT: %[[WHILE:.*]]:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
    // CHECK-NEXT:   cond {
    // CHECK-NEXT:   %[[C32:.*]] = sdy.constant dense<32>
    // CHECK-NEXT:   %[[COND:.*]] = mhlo.compare LT, %iterArg_0, %[[C32]]
    // CHECK-NEXT:   mhlo.return %[[COND]]
    // CHECK-NEXT: } do {
    // CHECK-NEXT:   %[[C1:.*]] = sdy.constant dense<1>
    // CHECK-NEXT:   %[[ADD_0:.*]] = mhlo.add %iterArg_0, %[[C1]]
    // CHECK-NEXT:   %[[ADD_1:.*]] = mhlo.add %iterArg, %iterArg
    // CHECK-NEXT:   mhlo.return %[[ADD_1]], %[[ADD_0]]
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[WHILE]]#0
    %0 = mhlo.constant dense<0> : tensor<i32>
    %1:2 = mhlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
      cond {
      %2 = mhlo.constant dense<32> : tensor<i32>
      %3 = mhlo.compare LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      mhlo.return %3 : tensor<i1>
    } do {
      %2 = mhlo.constant dense<1> : tensor<i32>
      %3 = mhlo.add %iterArg_0, %2 : tensor<i32>
      %4 = mhlo.add %iterArg, %iterArg : tensor<32x96xf32>
      mhlo.return %4, %3 : tensor<32x96xf32>, tensor<i32>
    }
    return %1#0 : tensor<32x96xf32>
  }

  // CHECK-LABEL: func @discard_shardings_on_unknown_ops(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p4]>}) {
  func.func @discard_shardings_on_unknown_ops(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22a\22}p0]>"}}
  ) -> tensor<32xi32> {
    // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %arg0, %arg0 : tensor<32xi32>
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}p2]> : tensor<32xi32>
    // CHECK-NEXT: %[[UNKNOWN:.*]] = mhlo.custom_call @UnknownCustomCall(%[[SHARDING]]) : (tensor<32xi32>) -> tensor<32xi32>
    // CHECK-NEXT: return %[[UNKNOWN]]
    %0 = mhlo.add %arg0, %arg0 {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p1]>]>"}} : tensor<32xi32>
    %1 = mhlo.custom_call @Sharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p2]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %2 = mhlo.custom_call @UnknownCustomCall(%1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p3]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%2) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\22a\22}p4]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %3 : tensor<32xi32>
  }
}
