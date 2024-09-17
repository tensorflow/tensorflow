// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-import 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

func.func @single_manual_comp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  // CHECK-NOT: xla.sdy.ManualComputation
  // CHECK:               %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a", "b"}
  // CHECK-SAME:              (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            %[[ADD_0:.*]] = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
  // CHECK-NEXT:            %[[DOT:.*]] = stablehlo.dot %[[ADD_0]], %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            %[[REDUCE:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  // CHECK-NEXT:            ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
  // CHECK-NEXT:              %[[ADD_1:.*]] = stablehlo.add %arg4, %arg5 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD_1]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            sdy.return %[[REDUCE]] : tensor<2x32xf32>
  // CHECK-NEXT:          } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:          return %[[MAN_COMP]] : tensor<8x32xf32>
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0, %arg1) {called_computations = [@local_xla.sdy.manual_computation_body], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {\\\22b\\\22}]>, <@mesh_0, [{\\\22b\\\22}, {}], replicated={\\\22a\\\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22, \\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}], replicated={\\\22b\\\22}>]>"}} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

func.func @manual_comp_using_another(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NOT: xla.sdy.ManualComputation
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            sdy.return %arg1 : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          %[[MAN_COMP_1:.*]] = sdy.manual_computation(%[[MAN_COMP_0]])
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_0, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_0, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"b"}
  // CHECK-SAME:              (%arg1: tensor<8x4xf32>) {
  // CHECK-NEXT:            sdy.return %arg1 : tensor<8x4xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_1]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_0], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\\\22a\\\22}, {}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%0) {called_computations = [@local_xla.sdy.manual_computation_body_1], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\\\22b\\\22}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_3(
func.func @local_xla.sdy.manual_computation_body_3(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_2], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>"}} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_2(
func.func @local_xla.sdy.manual_computation_body_2(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @nested_shmaps(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: xla.sdy.ManualComputation
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            %[[MAN_COMP_1:.*]] = sdy.manual_computation(%arg1)
  // CHECK-SAME{LITERAL}:       in_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       out_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       manual_axes={"b"}
  // CHECK-SAME:                (%arg2: tensor<2x4xf32>) {
  // CHECK-NEXT:              %[[MULT:.*]] = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
  // CHECK-NEXT:              sdy.return %[[MULT]] : tensor<2x4xf32>
  // CHECK-NEXT:            } : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT:            sdy.return %[[MAN_COMP_1]] : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_0]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_3], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>"}} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NOT: xla.sdy.ManualComputation
  // CHECK:               %[[MAN_COMP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh_1, [{"a"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg1: tensor<2x8xf32>) {
  // CHECK-NEXT:            %[[MAN_COMP_1:.*]] = sdy.manual_computation(%arg1)
  // CHECK-SAME{LITERAL}:       in_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       out_shardings=[<@mesh_1, [{}, {"b"}]>]
  // CHECK-SAME{LITERAL}:       manual_axes={"b"}
  // CHECK-SAME:                (%arg2: tensor<2x4xf32>) {
  // CHECK-NEXT:              %[[MULT:.*]] = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
  // CHECK-NEXT:              sdy.return %[[MULT]] : tensor<2x4xf32>
  // CHECK-NEXT:            } : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %[[MAN_COMP_1]], %[[MAN_COMP_1]] : tensor<2x8xf32>
  // CHECK-NEXT:            sdy.return %[[ADD]] : tensor<2x8xf32>
  // CHECK-NEXT:          } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:          return %[[MAN_COMP_0]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_5], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\\\22a\\\22}, {}]>]>"}} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body(
func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<2x8xf32>
  %1 = stablehlo.dot %0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = "stablehlo.all_reduce"(%1) <{replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %3 : tensor<f32>
  }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  return %2 : tensor<2x32xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_0(
func.func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_1(
func.func @local_xla.sdy.manual_computation_body_1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  return %arg0 : tensor<8x4xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_4(
func.func @local_xla.sdy.manual_computation_body_4(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-NOT: func @local_xla.sdy.manual_computation_body_5(
func.func @local_xla.sdy.manual_computation_body_5(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_4], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22b\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\\\22b\\\22}]>]>"}} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<2x8xf32>
  return %1 : tensor<2x8xf32>
}
