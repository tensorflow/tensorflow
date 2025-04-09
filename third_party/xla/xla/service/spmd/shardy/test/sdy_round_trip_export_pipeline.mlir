// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-export-pipeline 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["axis_0"=2, "axis_1"=4, "axis_2"=4]>
sdy.mesh @mesh_1 = <["axis_0"=16]>
sdy.mesh @mesh_2 = <["x"=8, "y"=4]>

// CHECK-NOT: sdy.mesh @mesh

// CHECK: module attributes {mhlo.frontend_attributes = {
// CHECK-SAME: xla.sdy.meshes = "{
// CHECK-SAME: mesh_0 = #sdy.mesh<[\\\22axis_0\\\22=2, \\\22axis_1\\\22=4, \\\22axis_2\\\22=4]>,
// CHECK-SAME: mesh_1 = #sdy.mesh<[\\\22axis_0\\\22=16]>,
// CHECK-SAME: mesh_2 = #sdy.mesh<[\\\22x\\\22=8, \\\22y\\\22=4]>}"}} {

// CHECK-LABEL: func @multiple_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{\\\22axis_2\\\22}, {\\\22axis_0\\\22, \\\22axis_1\\\22}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{}, {\\\22axis_0\\\22, \\\22axis_2\\\22}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{}, {\\\22axis_1\\\22}]>"}, mhlo.sharding =
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @multiple_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                              %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_0", "axis_2"}]>},
                              %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_1"}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: stablehlo.add
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22axis_1\\\22, \\\22axis_0\\\22}, {}]>]>"}, mhlo.sharding =
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_1","axis_0"}, {}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @multi_result_op
func.func @multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: stablehlo.reduce
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{}, {\\\22y\\\22}]>, <@mesh_2, [{\\\22y\\\22}, {}]>]>"}, mhlo.sharding =
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"y"}]>, <@mesh_2, [{"y"}, {}]>]>} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @split_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_2, [{\\\22y\\\22}, {\\\22x\\\22:(2)2}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg1: tensor<8x16xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_2, [{\\\22x\\\22:(1)2}, {\\\22x\\\22:(2)4}]>"}, mhlo.sharding =
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @split_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"y"}, {"x":(2)2}]>},
                      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x":(1)2}, {"x":(2)4}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: stablehlo.dot
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22:(1)2, \\\22x\\\22:(4)2}, {}]>]>"}, mhlo.sharding =
  %1 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x":(1)2, "x":(4)2}, {}]>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_result_sharding_returning_func_arg(
func.func @func_result_sharding_returning_func_arg(
  // CHECK: %arg0: tensor<8x16xf32>) -> (tensor<8x16xf32> {mhlo.sharding =
  %arg0: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}) {
  // CHECK:      %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22, ?}, {\\\22y\\\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_result_sharding_returning_op_value(%arg0: tensor<8x16xf32>)
func.func @func_result_sharding_returning_op_value(%arg0: tensor<8x16xf32>)
  // CHECK-SAME: -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"},
  // CHECK-SAME:     tensor<8x16xf32> {mhlo.sharding = "{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
  // CHECK-SAME:     tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"},
  // CHECK-SAME:     tensor<8x16xf32> {mhlo.sharding = "{replicated}"}) {
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{?}, {"y"}p4]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {"y"}p1]>},
      tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY:.*]]:2 = stablehlo.custom_call @sdy_testonly(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22, \\\22y\\\22}, {}]>, <@mesh_2, [{\\\22y\\\22, \\\22x\\\22}, {}]>]>"}, mhlo.sharding =
  // CHECK-NEXT: %[[ADD_RESULT_SHARDING_0:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%[[ADD]]) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22, ?}, {\\\22y\\\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY_RES_SHARDING_0:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%[[TEST_ONLY]]#0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{?}, {\\\22y\\\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY_RES_SHARDING_1:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%[[TEST_ONLY]]#1) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22}, {\\\22y\\\22}p1]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ADD_RESULT_SHARDING_1:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%[[ADD]]) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{}, {}]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[ADD_RESULT_SHARDING_0]], %[[TEST_ONLY_RES_SHARDING_0]], %[[TEST_ONLY_RES_SHARDING_1]], %[[ADD_RESULT_SHARDING_1]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1:2 = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x","y"}, {}]>, <@mesh_2, [{"y","x"}, {}]>]>} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %0, %1#0, %1#1, %0 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @sharding_constraint
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22, ?}, {?}]>]>"}, mhlo.sharding =
  %0 = sdy.sharding_constraint %arg0 <@mesh_2, [{"x", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @export_sharding_group
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @export_sharding_group(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: stablehlo.custom_call @local_xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "12 : i64"}}
  sdy.sharding_group %arg0 group_id = 12:  tensor<8x8xf32>
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @export_propagation_barrier
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @export_propagation_barrier(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %0 = stablehlo.custom_call @local_xla.sdy.PropagationBarrier(%arg0) {mhlo.frontend_attributes = {xla.sdy.allowed_direction = "2 : i32"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @constant
func.func @constant() -> tensor<i32> {
  // CHECK-NEXT: %[[CONST:.*]] = stablehlo.constant dense<0>
  // CHECK-NEXT: return %[[CONST]]
  %0 = sdy.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @inlined_mesh(
// CHECK-SAME: %arg0: tensor<32xi32>
// CHECK-SAME:   {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\\\22a\\\22=2, \\\22b\\\22=2]>, [{\\\22a\\\22}]>"},
// CHECK-SAME:    mhlo.sharding = "{devices=[2,2]<=[4] last_tile_dim_replicate}"})
// CHECK-SAME: -> (tensor<32xi32> {mhlo.sharding = "{maximal device=5}"}) {
func.func @inlined_mesh(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}]>}
) -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[5]>, []>}) {
  // CHECK-NEXT: %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0)
  // CHECK-SAME:   mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\\\22c\\\22=4]>, [{\\\22c\\\22}]>]>"}, mhlo.sharding = "{devices=[4]<=[4]}"}
  // CHECK-NEXT: %[[RESULT_SHARDING:.*]] = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%[[SHARDING]])
  // CHECK-SAME:   mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[], device_ids=[5]>, []>]>"}
  // CHECK-NEXT: return %[[RESULT_SHARDING]]
  %0 = sdy.sharding_constraint %arg0 <mesh<["c"=4]>, [{"c"}]> : tensor<32xi32>
  return %0 : tensor<32xi32>
}

// CHECK-LABEL: func @op_sharding_rule
func.func @op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1) {mhlo.frontend_attributes = {xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>"}}
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
  return %0 : tensor<8x2xf64>
}

// CHECK-LABEL: func @sharding_and_op_sharding_rule
func.func @sharding_and_op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
  // CHECK: stablehlo.custom_call @foo(%arg0, %arg1) {mhlo.frontend_attributes =
  // CHECK-SAME: {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\\\22x\\\22}, {}]>]>"
  // CHECK-SAME: xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>"}
  %0 = stablehlo.custom_call @foo(%arg0, %arg1)
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>,
     sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>}
    : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
  return %0 : tensor<8x2xf64>
}

// CHECK-LABEL: func @while_with_no_sharding
func.func @while_with_no_sharding(
    %arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_1 = %[[C0]])
  // CHECK-NOT: mhlo.sharding
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    stablehlo.return %iterArg, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----

// CHECK-NOT: xla.sdy.meshes

// CHECK-LABEL: func @non_sdy_module(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,16]<=[32] last_tile_dim_replicate}"},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,4,2]<=[2,16]T(1,0) last_tile_dim_replicate}"})
// CHECK-SAME:  -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
func.func @non_sdy_module(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
                          %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,16]<=[32] last_tile_dim_replicate}"},
                          %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,4,2]<=[2,16]T(1,0) last_tile_dim_replicate}"})
    -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg1 {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"}
  // CHECK-NOT: xla.sdy.sharding
  // CHECK-NOT: xla.sdy.sharding_rule
  %0 = stablehlo.add %arg0, %arg1 {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
