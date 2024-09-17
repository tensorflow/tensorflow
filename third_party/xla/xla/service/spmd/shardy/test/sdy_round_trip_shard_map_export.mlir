// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK: %[[SHMAP:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0, %arg1)
  // CHECK-SAME: {called_computations = [@local_xla.sdy.manual_computation_body],
  // CHECK-SAME: mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {\\\22b\\\22}]>, <@mesh_0, [{\\\22b\\\22}, {}], replicated={\\\22a\\\22}>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}}
  // CHECK-SAME: : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[SHMAP]] : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) {
      replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
    } : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"b"}]>}) {
  // CHECK: %[[SHMAP_0:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0)
  // CHECK-SAME: {called_computations = [@local_xla.sdy.manual_computation_body_0],
  // CHECK-SAME: mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}]>]>"}}
  // CHECK-SAME: : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:  %[[SHMAP_1:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%[[SHMAP_0]])
  // CHECK-SAME: {called_computations = [@local_xla.sdy.manual_computation_body_1],
  // CHECK-SAME: mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\\\22b\\\22}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\\\22b\\\22}]>]>"}}
  // CHECK-SAME: : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"b"}]>] out_shardings=[<@mesh_0, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK: %[[SHMAP:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0)
  // CHECK-SAME: {called_computations = [@local_xla.sdy.manual_computation_body_3],
  // CHECK-SAME: mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>"}}
  // CHECK-SAME: : (tensor<4x8xf32>) -> tensor<4x8xf32
  // CHECK-NEXT: return %[[SHMAP]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    sdy.return %1 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK: %[[SHMAP:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0)
  // CHECK-SAME: {called_computations = [@local_xla.sdy.manual_computation_body_5],
  // CHECK-SAME: mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>"}} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SHMAP]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %3 = stablehlo.add %1, %1 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    stablehlo.dot
// CHECK-NEXT:    "stablehlo.all_reduce"

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:    return %arg0 : tensor<2x8xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    return %arg0 : tensor<8x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_2(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:    stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_3(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32
// CHECK-NEXT:   stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0)
// CHECK-SAME:     {called_computations = [@local_xla.sdy.manual_computation_body_2],
// CHECK-SAME:     mhlo.frontend_attributes = {
// CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>",
// CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>",
// CHECK-SAME:     xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>"}}
// CHECK-SAME:     : (tensor<2x8xf32>) -> tensor<2x8xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_4(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:    stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_5(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[SHMAP:.*]] = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0)
// CHECK-SAME:     {called_computations = [@local_xla.sdy.manual_computation_body_4],
// CHECK-SAME:     mhlo.frontend_attributes = {
// CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>",
// CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>",
// CHECK-SAME:     xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>"}}
// CHECK-SAME:     : (tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[SHMAP]], %[[SHMAP]] : tensor<2x8xf32>
// CHECK-NEXT:   return %[[ADD]] : tensor<2x8xf32>
