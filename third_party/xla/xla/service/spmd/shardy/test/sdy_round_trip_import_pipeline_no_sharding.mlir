// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-pipeline 2>&1 | FileCheck %s
// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-pipeline='enable-hlo-sharding-v3=true' 2>&1 | FileCheck %s


// CHECK-NOT: sdy.mesh @mesh

module @no_meshes_module {
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
  stablehlo.custom_call @xla.sdy.ShardingGroup(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "21 : i64"}} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @import_propagation_barrier_backward
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @import_propagation_barrier_backward(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK %r = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD :  tensor<8x8xf32>
  %r = stablehlo.custom_call @xla.sdy.PropagationBarrier(%arg0) {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.allowed_direction = "2 : i32"}} : (tensor<8x8xf32>) -> (tensor<8x8xf32>)
  return %r : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @import_propagation_barrier_forward
// CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @import_propagation_barrier_forward(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK %r = sdy.propagation_barrier %arg0 allowed_direction=FORWARD :  tensor<8x8xf32>
  %r = stablehlo.custom_call @xla.sdy.PropagationBarrier(%arg0) {mhlo.frontend_attributes = {xla.sdy.allowed_direction = "1 : i32"}} : (tensor<8x8xf32>) -> (tensor<8x8xf32>)
  return %r : tensor<8x8xf32>
}

// -----

func.func @callback_no_result(%arg0: tensor<f64>) {
  // CHECK:      %[[C:.*]] = sdy.constant
  // CHECK-NEXT: stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0) {
  // CHECK-SAME:   api_version = 2 : i32, backend_config = "56238273106176",
  // CHECK-SAME:   has_side_effect = true,
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>],
  // CHECK-SAME:   result_layouts = []
  // CHECK-SAME: } : (tensor<i64>, tensor<f64>) -> ()
  %c = stablehlo.constant dense<56238273106176> : tensor<i64>
  stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {api_version = 2 : i32, backend_config = "56238273106176", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = []} : (tensor<i64>, tensor<f64>) -> ()
  return
}

// -----
module @send_with_sdy_sharding_module {
  // CHECK: sdy.mesh @maximal_mesh_0
  sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
  // CHECK-LABEL: func.func @send_with_sdy_sharding
  func.func @send_with_sdy_sharding(%arg0: tensor<i32>,
                                     %arg1: !stablehlo.token) -> !stablehlo.token {
  // CHECK-NEXT: %0 = "stablehlo.send"(%arg0, %arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}> {mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_rendezvous = "_host_callback_dtoh_0"}, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>, xla_shape = "token[]"}
    %1 = "stablehlo.send"(%arg0, %arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true}>
      {mhlo.frontend_attributes = {_xla_host_transfer_handler_name = "tf_rendezvous", _xla_host_transfer_rendezvous = "_host_callback_dtoh_0"},
      sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>, xla_shape = "token[]"} : (tensor<i32>, !stablehlo.token) -> !stablehlo.token
    return %1 : !stablehlo.token
  }
}

// -----
// CHECK-LABEL: func @non_flat_call_graph_all_inlineable
// CHECK-NOT: sdy.named_computation
func.func @non_flat_call_graph_all_inlineable(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %0 = call @foo(%arg0)
  // CHECK: %1 = stablehlo.negate %0 : tensor<8xf32>
  // CHECK: %2 = call @baz(%1)
  // CHECK: return %2 : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 : tensor<8xf32>
  %2 = call @baz(%1) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  %1 = call @bar(%0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK: func private @bar
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: func private @baz
func.func private @baz(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----
// CHECK-LABEL: func @uninlineable_call
func.func @uninlineable_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: %0 = call @foo(%arg0)
  // CHECK: return %0 : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
