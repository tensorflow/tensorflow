// RUN: sdy_opt %s -xla-sdy-round-trip-export-pipeline 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <"axis_0"=2, "axis_1"=4, "axis_2"=4>
sdy.mesh @mesh_1 = <"axis_0"=16>
sdy.mesh @mesh_2 = <"x"=8, "y"=4>

// CHECK-NOT: sdy.mesh @mesh

// CHECK: module attributes {mhlo.frontend_attributes = {
// CHECK-SAME: xla.sdy.meshes = "{
// CHECK-SAME: mesh_0 = \22#sdy.mesh<\\22axis_0\\22=2, \\22axis_1\\22=4, \\22axis_2\\22=4>\22,
// CHECK-SAME: mesh_1 = \22#sdy.mesh<\\22axis_0\\22=16>\22,
// CHECK-SAME: mesh_2 = \22#sdy.mesh<\\22x\\22=8, \\22y\\22=4>\22}"}} {

// CHECK-LABEL: func @multiple_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{\22axis_2\22}, {\22axis_0\22, \22axis_1\22}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{}, {\22axis_0\22, \22axis_2\22}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_0, [{}, {\22axis_1\22}]>"}, mhlo.sharding =
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @multiple_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                              %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_0", "axis_2"}]>},
                              %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_1"}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: mhlo.add
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_0, [{\22axis_1\22, \22axis_0\22}, {}]>]>"}, mhlo.sharding =
  %0 = mhlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_1","axis_0"}, {}]>]>} : tensor<8x8xf32>
  %1 = "mhlo.dot" (%0, %arg2) : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @multi_result_op
func.func @multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: mhlo.reduce
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{}, {\22y\22}]>, <@mesh_2, [{\22y\22}, {}]>]>"}, mhlo.sharding =
  %1:2 = mhlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"y"}]>, <@mesh_2, [{"y"}, {}]>]>} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = mhlo.add %arg2, %arg4 : tensor<f32>
      %3 = mhlo.add %arg3, %arg5 : tensor<f32>
      mhlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @split_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_2, [{\22y\22}, {\22x\22:(2)2}]>"}, mhlo.sharding =
// CHECK-SAME:      %arg1: tensor<8x16xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_2, [{\22x\22:(1)2}, {\22x\22:(2)4}]>"}, mhlo.sharding =
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @split_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"y"}, {"x":(2)2}]>},
                      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x":(1)2}, {"x":(2)4}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: "mhlo.dot"
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22:(1)2, \22x\22:(4)2}, {}]>]>"}, mhlo.sharding =
  %1 = "mhlo.dot" (%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x":(1)2, "x":(4)2}, {}]>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_result_sharding_returning_func_arg(
func.func @func_result_sharding_returning_func_arg(
  // CHECK: %arg0: tensor<8x16xf32>) -> (tensor<8x16xf32> {mhlo.sharding =
  %arg0: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}) {
  // CHECK:      %[[CUSTOM_CALL:.*]] = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22, ?}, {\22y\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]] : tensor<8x16xf32>
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_result_sharding_returning_op_value(
func.func @func_result_sharding_returning_op_value(
  // CHECK: %arg0: tensor<8x16xf32>)
  // CHECK-SAME: -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}, tensor<8x16xf32> {mhlo.sharding = "{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"}, tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
  %arg0: tensor<8x16xf32>) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", ?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{?}, {"y"}p4]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {"y"}p1]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY:.*]]:2 = mhlo.custom_call @sdy_testonly(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22, \22y\22}, {}]>, <@mesh_2, [{\22y\22, \22x\22}, {}]>]>"}, mhlo.sharding =
  // CHECK-NEXT: %[[ADD_RESULT_SHARDING:.*]] = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%[[ADD]]) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22, ?}, {\22y\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY_RES_SHARDING_0:.*]] = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%[[TEST_ONLY]]#0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{?}, {\22y\22}p4]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[TEST_ONLY_RES_SHARDING_1:.*]] = mhlo.custom_call @local_xla.sdy.FuncResultSharding(%[[TEST_ONLY]]#1) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22}, {\22y\22}p1]>]>"}} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[ADD_RESULT_SHARDING]], %[[TEST_ONLY_RES_SHARDING_0]], %[[TEST_ONLY_RES_SHARDING_1]] : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
  %0 = mhlo.add %arg0, %arg0 : tensor<8x16xf32>
  %1:2 = mhlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x","y"}, {}]>, <@mesh_2, [{"y","x"}, {}]>]>} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %0, %1#0, %1#1 : tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @sharding_constraint
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: mhlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_2, [{\22x\22, ?}, {?}]>]>"}, mhlo.sharding =
  %0 = sdy.sharding_constraint %arg0 <@mesh_2, [{"x", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @constant
func.func @constant() -> tensor<i32> {
  // CHECK-NEXT: %[[CONST:.*]] = mhlo.constant dense<0>
  // CHECK-NEXT: return %[[CONST]]
  %0 = sdy.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}
