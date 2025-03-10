// RUN: sdy_opt %s --split-input-file -xla-sdy-import-constants -xla-sdy-round-trip-import-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: module @multiple_func_result_shardings
module @multiple_func_result_shardings attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
    "{mesh = #sdy.mesh<[\\\22a\\\22=8, \\\22b\\\22=8, \\\22c\\\22=8]>, mesh2 = #sdy.mesh<[\\\22a\\\22=1, \\\22b\\\22=4, \\\22c\\\22=1]>}"}} {
  // CHECK: sdy.mesh @mesh = <["a"=8, "b"=8, "c"=8]>

  // CHECK: sdy.mesh @mesh2 = <["a"=1, "b"=4, "c"=1]>

  // CHECK-LABEL: func @func_results_with_sharding
  // CHECK-SAME:    %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME:    %arg1: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p1]>},
  // CHECK-SAME:    %arg2: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}p0]>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p1]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}p0]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p3]>}) {
  // CHECK-NEXT:   return %arg0, %arg1, %arg0, %arg1, %arg1, %arg2
  // CHECK-NEXT: }
  func.func @func_results_with_sharding(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22b\\\22}p2]>"}},
    %arg1: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22a\\\22}p1]>"}},
    %arg2: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22c\\\22}p0]>"}}
  ) -> (tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>) {
    %0 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p0]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22b\\\22}p2]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %2 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p1]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22c\\\22}p0]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %4 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg2) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p3]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %0, %1, %2, %3, %1, %4 : tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>
  }

  // This might happen due to inlined funcs that originally had result shardings
  // CHECK-LABEL: func @func_result_shardings_used_by_other_ops(
  // CHECK-SAME:    %arg0: tensor<32xi32>, %arg1: tensor<32xi32>
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}p2]>},
  // CHECK-SAME:    tensor<32xi32>) {
  // CHECK-NEXT:   %[[ADD:.*]] =  stablehlo.add %arg0, %arg1
  // CHECK-NEXT:   return %arg0, %[[ADD]]
  // CHECK-NEXT: }
  func.func @func_result_shardings_used_by_other_ops(
    %arg0: tensor<32xi32>, %arg1: tensor<32xi32>
  ) -> (tensor<32xi32>, tensor<32xi32>) {
    %0 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p0]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22b\\\22}p2]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %2 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p3]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = stablehlo.add %1, %2 : tensor<32xi32>
    return %1, %3 : tensor<32xi32>, tensor<32xi32>
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
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.constant dense<32> : tensor<i32>
    %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
      cond {
      %4 = stablehlo.compare LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
      %5 = stablehlo.add %iterArg, %arg1 : tensor<32x96xf32>
      stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
    }
    return %3#0 : tensor<32x96xf32>
  }

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
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
      cond {
      %2 = stablehlo.constant dense<32> : tensor<i32>
      %3 = stablehlo.compare LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    } do {
      %2 = stablehlo.constant dense<1> : tensor<i32>
      %3 = stablehlo.add %iterArg_0, %2 : tensor<i32>
      %4 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
      stablehlo.return %4, %3 : tensor<32x96xf32>, tensor<i32>
    }
    return %1#0 : tensor<32x96xf32>
  }

  // CHECK-LABEL: func @discard_shardings_on_unknown_ops(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p0]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}p4]>}) {
  func.func @discard_shardings_on_unknown_ops(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22a\\\22}p0]>"}}
  ) -> tensor<32xi32> {
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 : tensor<32xi32>
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}p2]> : tensor<32xi32>
    // CHECK-NEXT: %[[UNKNOWN:.*]] = stablehlo.custom_call @UnknownCustomCall(%[[SHARDING]]) : (tensor<32xi32>) -> tensor<32xi32>
    // CHECK-NEXT: return %[[UNKNOWN]]
    %0 = stablehlo.add %arg0, %arg0 {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p1]>]>"}} : tensor<32xi32>
    %1 = stablehlo.custom_call @Sharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p2]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %2 = stablehlo.custom_call @UnknownCustomCall(%1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p3]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%2) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}p4]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %3 : tensor<32xi32>
  }

  // CHECK-LABEL: func @inlined_mesh(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<mesh<[], device_ids=[5]>, []>}) {
  func.func @inlined_mesh(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\\\22a\\\22=2, \\\22b\\\22=2]>, [{\\\22a\\\22}]>"}}
  ) -> tensor<32xi32> {
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %arg0 <mesh<["c"=4]>, [{"c"}]> : tensor<32xi32>
    // CHECK-NEXT: return %[[SHARDING]]
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\\\22c\\\22=4]>, [{\\\22c\\\22}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[], device_ids=[5]>, []>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %1 : tensor<32xi32>
  }

  // CHECK-LABEL: func @shardings_with_size_one_axes
  // CHECK-SAME:    %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh2, [{"b"}p1]>},
  // CHECK-SAME:    %arg1: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh2, [{}], replicated={"b"}>},
  // CHECK-SAME:    %arg2: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh2, [{"b", ?}p0]>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh2, [{}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh2, [{"b"}]>}) {
  func.func @shardings_with_size_one_axes(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh2, [{\\\22b\\\22}p1], replicated={\\\22c\\\22}>"}},
    %arg1: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh2, [{\\\22a\\\22}p2], replicated={\\\22b\\\22}>"}},
    %arg2: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh2, [{\\\22c\\\22, \\\22b\\\22, ?}p0]>"}}
  ) -> (tensor<32xi32>, tensor<32xi32>) {
    // CHECK-NEXT:   %[[SC1:.*]] = sdy.sharding_constraint %arg0 <@mesh2, [{"b", ?}]>
    // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[SC1]], %[[SC1]]
    // CHECK-NOT:    sdy.sharding
    // CHECK-NEXT:   %[[SC2:.*]] = sdy.sharding_constraint %arg1 <@mesh2, [{}]>
    // CHECK-NEXT:   return %[[ADD]], %[[SC2]]
    // CHECK-NEXT: }
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh2, [{\\\22a\\\22, \\\22b\\\22, ?}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.add %0, %0 : tensor<32xi32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh2, [{\\\22c\\\22, \\\22a\\\22}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %3 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%1) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh2, [{\\\22a\\\22}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %4 = stablehlo.custom_call @local_xla.sdy.FuncResultSharding(%2) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh2, [{\\\22b\\\22}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %3, %4 : tensor<32xi32>, tensor<32xi32>
  }

  // CHECK-LABEL: func @manual_computation_with_size_one_axes
  func.func @manual_computation_with_size_one_axes(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
    // CHECK-NOT: call @local_xla.sdy.manual_computation_body
    // CHECK:               %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
    // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh2, [{}, {"b"}]>, <@mesh2, [{}, {"b"}]>]
    // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh2, [{}, {"b"}]>]
    // CHECK-SAME{LITERAL}:     manual_axes={"b"}
    // CHECK-SAME:              (%arg2: tensor<16x8xf32>, %arg3: tensor<16x8xf32>) {
    // CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %arg2, %arg3
    // CHECK-NEXT:            sdy.return %[[ADD]]
    // CHECK-NEXT:          } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
    // CHECK-NEXT:          return %[[MAN_COMP]]
    %0:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1) : (tensor<16x32xf32>, tensor<16x32xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>)
    %1 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh2, [{\\\22a\\\22}, {\\\22b\\\22}]>, <@mesh2, [{}, {\\\22b\\\22}], replicated={\\\22a\\\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh2, [{}, {\\\22b\\\22, \\\22a\\\22}]>]>"}} : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
    %2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1) : (tensor<16x8xf32>) -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }

  // CHECK-NOT: func @local_xla.sdy.manual_computation_body(
  func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<16x8xf32>, %arg1: tensor<16x8xf32>) -> tensor<16x8xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}

// -----

// CHECK-NOT: sdy.mesh @mesh

module @no_meshes_module attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{}"}} {
  // CHECK-LABEL: func @no_sharding_rule
  func.func @no_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
    // CHECK-NEXT: stablehlo.custom_call @foo(%arg0, %arg1) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
    %0 = stablehlo.custom_call @foo(%arg0, %arg1) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
   return %0 : tensor<8x2xf64>
  }

  // CHECK-LABEL: func @op_sharding_rule
  func.func @op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
    // CHECK-NEXT: stablehlo.custom_call @foo(%arg0, %arg1) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>}
    %0 = stablehlo.custom_call @foo(%arg0, %arg1)
      {mhlo.frontend_attributes = {xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>"}} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
    return %0 : tensor<8x2xf64>
  }
}

// -----

// CHECK-NOT: sdy.mesh @mesh

module @no_meshes_attr_module {
  // CHECK-LABEL: func @op_sharding_rule
  func.func @op_sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf64> {
    // CHECK-NEXT: stablehlo.custom_call @foo(%arg0, %arg1) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>}
    %0 = stablehlo.custom_call @foo(%arg0, %arg1)
      {mhlo.frontend_attributes = {xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>"}} : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf64>
    return %0 : tensor<8x2xf64>
  }
}

// -----

// CHECK-LABEL: func @import_sharding_group
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @import_sharding_group(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK sdy.sharding_group %arg0 group_id = 21:  tensor<8x8xf32>
  stablehlo.custom_call @local_xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "21 : i64"}} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

func.func @callback_no_result(%arg0: tensor<f64>) {
  // CHECK:      %[[C:.*]] = sdy.constant
  // CHECK-NEXT: stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0) {
  // CHECK-SAME:   api_version = 2 : i32, backend_config = "56238273106176",
  // CHECK-SAME:   has_side_effect = true,
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>],
  // CHECK-SAME:   result_layouts = [dense<> : tensor<0xindex>]
  // CHECK-SAME: } : (tensor<i64>, tensor<f64>) -> tensor<i64>
  %c = stablehlo.constant dense<56238273106176> : tensor<i64>
  stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {api_version = 2 : i32, backend_config = "56238273106176", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = []} : (tensor<i64>, tensor<f64>) -> ()
  return
}

// -----

module @maximal_sharding_module attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{maximal_mesh_0 = #sdy.mesh<[], device_ids=[0]>}"}} {
  // CHECK-LABEL: @maximal_sharding_empty_tuple
  func.func @maximal_sharding_empty_tuple(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    // CHECK-NEXT: %[[DUMMY_VAL:.*]] = stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {
    // CHECK-SAME:   api_version = 4 : i32, backend_config = {descriptor = 126001424235520 : ui64},
    // CHECK-SAME:   has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>],
    // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>, xla_shape = "()"
    // CHECK-SAME: } : (tensor<2xi64>) -> tensor<2xi64>
    // CHECK-NEXT: return %arg0 : tensor<2xi64>
    %2 = stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {
      api_version = 4 : i32, backend_config = {descriptor = 126001424235520 : ui64},
      has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@maximal_mesh_0, []>]>"},
      mhlo.sharding = "{{maximal device=0}}",
      operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [], xla_shape = "()"
    } : (tensor<2xi64>) -> tuple<>
    return %arg0 : tensor<2xi64>
  }
}
