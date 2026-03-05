// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-pipeline='enable-hlo-sharding-v3=true' 2>&1 | FileCheck %s

// CHECK-LABEL: module @module_1
module @module_1 {
  // CHECK: sdy.mesh @mesh = <["a"=8, "b"=8, "c"=8]>
  // CHECK: sdy.mesh @mesh_0 = <["a"=2, "b"=2]>
  // CHECK: sdy.mesh @maximal_mesh_5 = <[], device_ids=[5]>
  // CHECK: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

  // CHECK-LABEL: func @results_with_sharding
  // CHECK-SAME:    %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>},
  // CHECK-SAME:    %arg1: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:    %arg2: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}]>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>},
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  func.func @results_with_sharding(
    %arg0: tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{b}]}"},
    %arg1: tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{a}]}"},
    %arg2: tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{c}]}"}
  ) -> (tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{a}]}"},
        tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{b}]}"},
        tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{a}]}"},
        tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{c}]}"},
        tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{b}]}"},
        tensor<32xi32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{a}]}"}) {
    // CHECK-NEXT: return %arg0, %arg1, %arg0, %arg1, %arg1, %arg2
    return %arg0, %arg1, %arg0, %arg1, %arg1, %arg2 : tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>, tensor<32xi32>
  }

  // CHECK-LABEL: func @while_with_free_variables
  func.func @while_with_free_variables(
      %arg0: tensor<32x96xf32>,
      %arg1: tensor<32x96xf32> {mhlo.sharding = "{mesh[a=8,b=8,c=8], [{?}, {?}]}"})
      -> tensor<32x96xf32> {
    // CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0>
    // CHECK-NEXT: %[[C1:.*]] = sdy.constant dense<1>
    // CHECK-NEXT: %[[C32:.*]] = sdy.constant dense<32>
    // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{?}, {?}]>
    // CHECK-NEXT: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]])
    // CHECK-NEXT:   cond {
    // CHECK-NEXT:   %[[COND:.*]] = stablehlo.compare  LT, %iterArg_0, %[[C32]]
    // CHECK-NEXT:   stablehlo.return %[[COND]]
    // CHECK-DAG:   %[[ADD_0:.*]] = stablehlo.add %iterArg_0, %[[C1]]
    // CHECK-DAG:   %[[ADD_1:.*]] = stablehlo.add %iterArg, %[[SC]]
    // CHECK-NEXT:   stablehlo.return %[[ADD_1]], %[[ADD_0]]
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[WHILE]]#0
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.constant dense<32> : tensor<i32>
    %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
      cond {
      %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
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

  // Test that inlined meshes are lifted and deduplicated.

  // CHECK-LABEL: func @inlined_mesh(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@maximal_mesh_5, []>}) {
  func.func @inlined_mesh(
    %arg0: tensor<32xi32> {mhlo.sharding = "{mesh[a=2,b=2], [{a}]}"}
  ) -> (tensor<32xi32> {mhlo.sharding = "{maximal_mesh[device_id=5]}"}) {
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %arg0 <@mesh_0, [{"a", "b"}]> : tensor<32xi32>
    // CHECK-NEXT: return %[[SHARDING]]
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{mesh[c=4], [{c}]}"} : (tensor<32xi32>) -> tensor<32xi32>
    return %0 : tensor<32xi32>
  }

  // CHECK-LABEL: func @manual_computation_nested_tuples
  func.func @manual_computation_nested_tuples(%arg0: tensor<8xi64>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK-NEXT: %[[SPLIT_LOW:.*]] = stablehlo.custom_call @X64SplitLow(%arg0)
    // CHECK-NEXT: %[[SPLIT_HIGH:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0)
    // CHECK-NEXT: %[[MAN_COMP:.*]] = sdy.manual_computation(%[[SPLIT_LOW]], %[[SPLIT_HIGH]], %arg1)
    // CHECK-SAME:     in_shardings=[<@mesh, [{"a"}]>, <@mesh, [{"a"}]>, <@mesh, [{"a"}]>]
    // CHECK-SAME:     out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"}
    // CHECK-SAME:     (%arg2: tensor<1xui32>, %arg3: tensor<1xui32>, %arg4: tensor<1xi32>) {
    // CHECK-NEXT:   %[[CONVERT:.*]] = stablehlo.convert %arg2
    // CHECK-NEXT:   %[[SUB:.*]] = stablehlo.subtract %[[CONVERT]], %arg4
    // CHECK-NEXT:   sdy.return %[[SUB]]
    // CHECK-NEXT: }
    // CHECK-NEXT: return %[[MAN_COMP]]
    %0 = stablehlo.custom_call @X64SplitLow(%arg0) : (tensor<8xi64>) -> tensor<8xui32>
    %1 = stablehlo.custom_call @X64SplitHigh(%arg0) : (tensor<8xi64>) -> tensor<8xui32>
    %2 = stablehlo.tuple %0, %1 : tuple<tensor<8xui32>, tensor<8xui32>>
    %3 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%2, %arg1) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>, <@mesh, [{\"a\"}]>, <@mesh, [{\"a\"}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\"a\"}>"}} : (tuple<tensor<8xui32>, tensor<8xui32>>, tensor<8xi32>) -> tuple<tuple<tensor<1xui32>, tensor<1xui32>>, tensor<1xi32>>
    %4 = stablehlo.get_tuple_element %3[0] : (tuple<tuple<tensor<1xui32>, tensor<1xui32>>, tensor<1xi32>>) -> tuple<tensor<1xui32>, tensor<1xui32>>
    %5 = stablehlo.get_tuple_element %3[1] : (tuple<tuple<tensor<1xui32>, tensor<1xui32>>, tensor<1xi32>>) -> tensor<1xi32>
    %6 = stablehlo.get_tuple_element %4[0] : (tuple<tensor<1xui32>, tensor<1xui32>>) -> tensor<1xui32>
    %7 = stablehlo.get_tuple_element %4[1] : (tuple<tensor<1xui32>, tensor<1xui32>>) -> tensor<1xui32>
    %8 = call @xla.sdy.manual_computation_body(%6, %7, %5) : (tensor<1xui32>, tensor<1xui32>, tensor<1xi32>) -> tensor<1xi32>
    %9 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%8) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\"a\"}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}]>]>"}} : (tensor<1xi32>) -> tensor<8xi32>
    return %9 : tensor<8xi32>
  }

  func.func private @xla.sdy.manual_computation_body(%arg0: tensor<1xui32>, %arg1: tensor<1xui32>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<1xui32>) -> tensor<1xi32>
    %1 = stablehlo.subtract %0, %arg2 : tensor<1xi32>
    return %1 : tensor<1xi32>
  }

  // CHECK-LABEL: func @frontend_attr_not_sharding
  // CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", "b"}, {?}]>},
  // CHECK-SAME:    %arg1: tensor<16x8xf32> {mhlo.frontend_attributes = {baz = 1 : i32, foo = "bar"}},
  // CHECK-SAME:    %arg2: !stablehlo.token) -> tensor<16x8xf32> {
  func.func @frontend_attr_not_sharding(
    %arg0: tensor<16x8xf32> {mhlo.sharding = "{mesh[a=1,b=4,c=1], [{b},{?}]}"},
    %arg1: tensor<16x8xf32> {mhlo.frontend_attributes = {baz = 1 : i32, foo = "bar"}},
    %arg2: !stablehlo.token) -> tensor<16x8xf32> {
    // CHECK-NEXT: %[[SEND:.*]] = "stablehlo.send"(%arg0, %arg2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}> {mhlo.frontend_attributes = {baz = 1 : i32}, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (tensor<16x8xf32>, !stablehlo.token) -> !stablehlo.token
    // CHECK-NEXT: %[[RECV:.*]]:2 = "stablehlo.recv"(%[[SEND]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>, is_host_transfer = true}> {mhlo.frontend_attributes = {baz = 1 : i32}, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>, <@maximal_mesh_0, []>]>} : (!stablehlo.token) -> (tensor<16x8xf32>, !stablehlo.token)
    // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RECV]]#0, %arg1 : tensor<16x8xf32>
    // CHECK-NEXT: return %[[ADD]] : tensor<16x8xf32>
    %0 = "stablehlo.send"(%arg0, %arg2) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
      is_host_transfer = true,
      mhlo.frontend_attributes = {baz = 1 : i32},
      mhlo.sharding = "{maximal_mesh[device_id=0]}"
    } : (tensor<16x8xf32>, !stablehlo.token) -> !stablehlo.token
    %1:2 = "stablehlo.recv"(%0) {
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 3>,
      is_host_transfer = true,
      mhlo.frontend_attributes = {baz = 1 : i32},
      mhlo.sharding = "{{maximal_mesh[device_id=0]}, {maximal_mesh[device_id=0]}}"
    } : (!stablehlo.token) -> (tensor<16x8xf32>, !stablehlo.token)
    %2 = stablehlo.add %1#0, %arg1 : tensor<16x8xf32>
    return %2 : tensor<16x8xf32>
  }
}

// -----

module @maximal_sharding_module {
  // CHECK-LABEL: @maximal_sharding_empty_tuple
  func.func @maximal_sharding_empty_tuple(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    // CHECK-NEXT: stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {
    // CHECK-SAME:   api_version = 4 : i32, backend_config = {descriptor = 126001424235520 : ui64},
    // CHECK-SAME:   has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [],
    // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>, xla_shape = "()"
    // CHECK-SAME: } : (tensor<2xi64>) -> ()
    // CHECK-NEXT: return %arg0 : tensor<2xi64>
    %2 = stablehlo.custom_call @xla_ffi_python_cpu_callback(%arg0) {
      api_version = 4 : i32, backend_config = {descriptor = 126001424235520 : ui64},
      has_side_effect = true,
      mhlo.sharding = "{maximal_mesh[device_id=0]}",
      operand_layouts = [dense<0> : tensor<1xindex>], result_layouts = [], xla_shape = "()"
    } : (tensor<2xi64>) -> tuple<>
    return %arg0 : tensor<2xi64>
  }
}

// -----

// CHECK-LABEL: module @main_func_in_out_tuple_shardings
// CHECK-NOT: xla.sdy.tuple_args_shardings
// CHECK-NOT: xla.sdy.tuple_results_shardings
module @main_func_in_out_tuple_shardings attributes {mhlo.frontend_attributes = {
  xla.sdy.tuple_args_shardings = "#sdy.sharding_per_value<[<mesh<[\"a\"=8, \"b\"=8, \"c\"=8]>, [{\"a\"}]>, <mesh<[\"a\"=8, \"b\"=8, \"c\"=8]>, [{\"b\"}]>]>",
  xla.sdy.use_tuple_args = "true",
  xla.sdy.tuple_results_shardings = "#sdy.sharding_per_value<[<mesh<[\"a\"=8, \"b\"=8, \"c\"=8]>, [{\"c\"}]>]>"}} {
  // CHECK: sdy.mesh @mesh = <["a"=8, "b"=8, "c"=8]>
  // CHECK-LABEL: func @main(
  // CHECK-SAME:    %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  // CHECK-SAME:    %arg1: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}
  // CHECK-SAME:  ) -> (
  // CHECK-SAME:    tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}]>}
  // CHECK-SAME:  ) {
  func.func @main(%arg0: tensor<32xi32>, %arg1: tensor<32xi32>) -> tensor<32xi32> {
    // CHECK-NEXT: return %arg0 : tensor<32xi32>
    return %arg0 : tensor<32xi32>
  }

  // CHECK-LABEL: func @non_main_func(
  // CHECK-SAME:    %arg0: tensor<32xi32>, %arg1: tensor<32xi32>) -> tensor<32xi32> {
  func.func @non_main_func(%arg0: tensor<32xi32>, %arg1: tensor<32xi32>) -> tensor<32xi32> {
    // CHECK-NEXT: return %arg0 : tensor<32xi32>
    return %arg0 : tensor<32xi32>
  }
}
