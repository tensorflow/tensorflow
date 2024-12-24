// RUN: sdy_opt %s -xla-sdy-mhlo-import-pipeline -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: sdy.mesh @mesh = <["axis_0"=8, "axis_1"=4]>

// CHECK-LABEL: func @sharding_custom_call_no_unspecified_dims(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_1"}, {"axis_0"}]>})
func.func @sharding_custom_call_no_unspecified_dims(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"}) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.sharding_constraint %arg0 <@mesh, [{"axis_0"}, {}]>
  %0 = mhlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_custom_call_with_unspecified_dims(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_1"}, {"axis_0"}]>})
func.func @sharding_custom_call_with_unspecified_dims(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"}) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.sharding_constraint %arg0 <@mesh, [{"axis_0"}, {?}]>
  %0 = mhlo.custom_call @Sharding(%arg0) {backend_config = "unspecified_dims=[1]", mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["axis_0"=4, "axis_1"=2]>

// CHECK-LABEL: func @manual(
// CHECK-SAME:       %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
// CHECK-SAME:       %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis_0"}, {}]>})
// CHECK-SAME:    -> tensor<8x8xf32> {
func.func @manual(%arg0: tensor<8x8xf32> {mhlo.sharding = "{replicated}"},
                  %arg1: tensor<4x8xf32> {mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"}) -> (tensor<8x8xf32>) {
  // CHECK:        sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"axis_0", "axis_1"}, {}]>, <@mesh, [{"axis_0"}, {}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"axis_0", "axis_1"}, {}]>]
  // CHECK-SAME:     manual_axes={"axis_0", "axis_1"} (%arg2: tensor<1x8xf32>, %arg3: tensor<1x8xf32>) {
  // CHECK-LABEL:  stablehlo.add
  // CHECK-LABEL:  sdy.return
  %0 = mhlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = mhlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<8x8xf32>) -> tensor<1x8xf32>
  %2 = mhlo.custom_call @Sharding(%arg1) {mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %3 = mhlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{manual}"} : (tensor<4x8xf32>) -> tensor<1x8xf32>
  %4 = call @shmap_body(%1, %3) : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
  %5 = mhlo.custom_call @Sharding(%4) {mhlo.sharding = "{manual}"} : (tensor<1x8xf32>) -> tensor<1x8xf32>
  %6 = mhlo.custom_call @SPMDShardToFullShape(%5) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1x8xf32>) -> tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// CHECK-NOT func.func @shmap_body
func.func @shmap_body(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>) -> (tensor<1x8xf32>) {
  %0 = mhlo.add %arg0, %arg1 : tensor<1x8xf32>
  return %0 : tensor<1x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["axis_0"=2, "axis_1"=2]>

// CHECK-LABEL: func @while_with_free_variables
func.func @while_with_free_variables(
    %arg0: tensor<32x96xf32>,
    %arg1: tensor<32x96xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"})
    -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
  // CHECK-NEXT: %[[C1:.*]] = sdy.constant dense<1>
  // CHECK-NEXT: %[[C32:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>]>} dense<32>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[ADD_0:.*]] = stablehlo.add %iterArg_0, %[[C1]]
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg, %[[SC]]
  // CHECK-NEXT:   stablehlo.return %[[ADD_1]], %[[ADD_0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#0
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<1> : tensor<i32>
  %2 = mhlo.constant {mhlo.sharding = "{replicated}"} dense<32> : tensor<i32>
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

// -----

// CHECK-LABEL: func @while_with_sinked_constants
func.func @while_with_sinked_constants(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
  // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
  // CHECK-NEXT:   cond {
  // CHECK-NEXT:   %[[C32:.*]] = sdy.constant dense<32>
  // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare LT, %iterArg_0, %[[C32]]
  // CHECK-NEXT:   stablehlo.return %[[COND]]
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %[[C1:.*]] = sdy.constant dense<1>
  // CHECK-NEXT:   %[[ADD_0:.*]] = stablehlo.add %iterArg_0, %[[C1]]
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %iterArg, %iterArg
  // CHECK-NEXT:   stablehlo.return %[[ADD_1]], %[[ADD_0]]
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

!tuple = tuple<tensor<8x8xf32>, tensor<4x8xf32>, tensor<8x16xf32>>

// CHECK-LABEL: func @custom_call_with_tuple_operand_result
func.func @custom_call_with_tuple_operand_result(%arg0: tensor<8x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[FOO:.*]]:3 = stablehlo.custom_call @foo(%arg0, %arg1, %arg2) :
  // CHECK-SAME:   (tensor<8x8xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
  // CHECK-SAME:   -> (tensor<8x8xf32>, tensor<4x8xf32>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[FOO]]#0
  %0 = mhlo.tuple %arg0, %arg1, %arg2 : !tuple
  %1 = mhlo.custom_call @foo(%0) : (!tuple) -> !tuple
  %2 = mhlo.get_tuple_element %1[0] : (!tuple) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @import_sharding_group_with_unused_result
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @import_sharding_group_with_unused_result(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK sdy.sharding_group %arg0 group_id = 21:  tensor<8x8xf32>
  %0 = mhlo.custom_call @local_xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "21 : i64"}} : (tensor<8x8xf32>) -> tuple<>
  return %arg0 : tensor<8x8xf32>
}
