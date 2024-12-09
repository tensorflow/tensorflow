// RUN: sdy_opt %s -xla-mhlo-round-trip-shard-map-import 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4]>
sdy.mesh @mesh_1 = <["a"=4, "b"=2]>
sdy.mesh @mesh_2 = <["a"=4, "b"=2, "c"=3]>

// CHECK-LABEL: func.func public @call_op_with_no_operands_or_results()
func.func public @call_op_with_no_operands_or_results() {
  // CHECK: sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
  // CHECK:   sdy.return
  // CHECK: } : () -> ()
  // CHECK: return
  call @shmap_body_empty() : () -> ()
  return
}
// CHECK-NOT: func.func private @shmap_body_empty
func.func private @shmap_body_empty() {
  return
}

// CHECK-LABEL: func.func public @call_op_with_one_operand_and_no_results
func.func public @call_op_with_one_operand_and_no_results(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK:               sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh_0, [{}], replicated={"a"}>] out_shardings=[] manual_axes={"a"} (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:            sdy.return
  // CHECK-NEXT:          } : (tensor<4xf32>) -> ()
  // CHECK-NEXT:          %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}], replicated={"a"}>]>} : (tensor<4xf32>) -> tensor<4xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<4xf32>) -> tensor<4xf32>
  call @shmap_body_one_argument_empty_body(%1) : (tensor<4xf32>) -> ()
  %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-NOT: func.func private @shmap_body_one_argument_empty_body
func.func private @shmap_body_one_argument_empty_body(%arg0: tensor<4xf32>) -> () {
  return
}

// CHECK-LABEL: func.func public @call_op_with_no_operands_and_one_result()
func.func public @call_op_with_no_operands_and_one_result() -> tensor<4xf32> {
  // CHECK:               %0 = sdy.manual_computation()
  // CHECK-SAME{LITERAL}:   in_shardings=[] out_shardings=[<@mesh_0, [{}], replicated={"a"}>] manual_axes={"a"} () {
  // CHECK-LABEL:           %cst = stablehlo.constant
  // CHECK-NEXT:            sdy.return %cst : tensor<4xf32>
  // CHECK-NEXT:          } : () -> tensor<4xf32>
  // CHECK-NEXT:          return %0 : tensor<4xf32>
  %0 = call @shmap_body_no_arg() : () -> (tensor<4xf32>)
  %1 = stablehlo.custom_call @Sharding(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = stablehlo.custom_call @SPMDShardToFullShape(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}], replicated={"a"}>]>} : (tensor<4xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
// CHECK-NOT: func.func private @shmap_body_no_arg()
func.func private @shmap_body_no_arg() -> tensor<4xf32> {
  %0 = stablehlo.constant dense <[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func public @call_op_with_shamp_body_in_middle
func.func public @call_op_with_shamp_body_in_middle(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg1 : tensor<4x32xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<4x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %0 : tensor<16x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = call @prefix_shmap_body_suffix(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @prefix_shmap_body_suffix(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_single_sharded_input_output_dim_0
func.func public @shard_map_single_sharded_input_output_dim_0(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg1 : tensor<4x32xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<4x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %0 : tensor<16x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = call @shmap_body(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_single_sharded_input_output_dim_1
func.func public @shard_map_single_sharded_input_output_dim_1(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {"a"}]>] out_shardings=[<@mesh_1, [{}, {"a"}]>] manual_axes={"a"} (%arg1: tensor<16x8xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg1 : tensor<16x8xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<16x8xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %0 : tensor<16x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"a"}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x8xf32>
  %2 = call @shmap_body_0(%1) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"a"}]>]>} : (tensor<16x8xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body_0
func.func private @shmap_body_0(%arg0: tensor<16x8xf32>) -> (tensor<16x8xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func.func public @shard_map_single_replicated_input_sharded_output
func.func public @shard_map_single_replicated_input_sharded_output(%arg0: tensor<16x32xf32>) -> tensor<16x256xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{}, {"a", "b"}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<16x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x256xf32>
  // CHECK-NEXT:     return %0 : tensor<16x256xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}], replicated={"a", "b"}>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2 = call @shmap_body_1(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"a", "b"}]>]>} : (tensor<16x32xf32>) -> tensor<16x256xf32>
  return %4 : tensor<16x256xf32>
}
// CHECK-NOT func.func private @shmap_body_1
func.func private @shmap_body_1(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_contracting_dim_matmul_all_reduce
func.func public @shard_map_contracting_dim_matmul_all_reduce(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK:               %0 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:      in_shardings=[<@mesh_1, [{"a"}, {"b"}]>, <@mesh_1, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_1, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            %2 = "stablehlo.all_reduce"(%1) <{
  // CHECK-SAME{LITERAL}:     channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids
  // CHECK-SAME:            }> ({
  // CHECK-NEXT:            ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
  // CHECK-NEXT:              %3 = stablehlo.add %arg4, %arg5 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %3 : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT:            sdy.return %2 : tensor<2x32xf32>
  // CHECK-NEXT:          } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:          return %0 : tensor<8x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {"b"}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<8x16xf32>) -> tensor<2x8xf32>
  %2 = stablehlo.custom_call @Sharding(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b"}, {}], replicated={"a"}>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) : (tensor<16x32xf32>) -> tensor<8x32xf32>
  %4 = call @shmap_body_2(%1, %3) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %5 = stablehlo.custom_call @Sharding(%4) : (tensor<2x32xf32>) -> tensor<2x32xf32>
  %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}], replicated={"b"}>]>}: (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %6 : tensor<8x32xf32>
}
// CHECK-NOT: func.func private @shmap_body_2
func.func private @shmap_body_2(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> (tensor<2x32xf32>) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %1 = "stablehlo.all_reduce"(%0) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids} : (tensor<2x32xf32>) -> tensor<2x32xf32>
  return %1 : tensor<2x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_wrong_callee_name
func.func public @shard_map_wrong_callee_name(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  // CHECK: call @shmap_head
  // CHECK-NOT: sdy.manual_computation
  %2 = call @shmap_head(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
// CHECK-LABEL: func.func private @shmap_head
func.func private @shmap_head(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_multiple_results
func.func public @shard_map_multiple_results(%arg0: tensor<16x32xf32>) -> tensor<128x32xf32> {
  // CHECK:          %[[SHARD_MAP:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{"a", "b"}, {}]>, <@mesh_1, [{"b", "a"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       sdy.return %arg1, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %[[SHARD_MAP]]#0, %[[SHARD_MAP]]#1 : tensor<128x32xf32>
  // CHECK-NEXT:     return %[[ADD]] : tensor<128x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}], replicated={"a", "b"}>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2:2 = call @shmap_body_4(%1) : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  %3 = stablehlo.custom_call @Sharding(%2#0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a", "b"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<128x32xf32>
  %5 = stablehlo.custom_call @Sharding(%2#1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b", "a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<128x32xf32>
  %7 = stablehlo.add %4, %6 : tensor<128x32xf32>
  return %7 : tensor<128x32xf32>
}
// CHECK-NOT: func.func private @shmap_body_4
func.func private @shmap_body_4(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  return %arg0, %arg0 : tensor<16x32xf32>, tensor<16x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_multiple_call_ops
func.func public @shard_map_multiple_call_ops(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) {
  // CHECK-NEXT:     %[[SHARD_MAP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:       sdy.return %[[ADD_0]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     %[[SHARD_MAP_1:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {"a"}]>] out_shardings=[<@mesh_1, [{}, {"a"}]>] manual_axes={"a"} (%arg1: tensor<16x8xf32>) {
  // CHECK-NEXT:       %[[MUL:.*]] = stablehlo.multiply %arg1, %arg1
  // CHECK-NEXT:       sdy.return %[[MUL]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     %[[SHARD_MAP_2:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:       sdy.return %[[ADD_1]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %[[SHARD_MAP_0]], %[[SHARD_MAP_1]], %[[SHARD_MAP_2]]
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = call @shmap_body_5(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>

  %5 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"a"}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %6 = stablehlo.custom_call @SPMDFullToShardShape(%5) : (tensor<16x32xf32>) -> tensor<16x8xf32>
  %7 = call @shmap_body_6(%6) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %8 = stablehlo.custom_call @Sharding(%7) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %9 = stablehlo.custom_call @SPMDShardToFullShape(%8) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"a"}]>]>} : (tensor<16x8xf32>) -> tensor<16x32xf32>

  %10 = call @shmap_body_5(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %11 = stablehlo.custom_call @Sharding(%10) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>

  return %4, %9, %12 : tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body_5(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body_6(%arg0: tensor<16x8xf32>) -> (tensor<16x8xf32>) {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func.func public @sharding_with_missing_manual_axes
func.func public @sharding_with_missing_manual_axes(%arg0: tensor<16x16xf32>) -> tensor<32x4xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_2, [{"b"}, {"a"}]>] out_shardings=[<@mesh_2, [{"a"}, {}], replicated={"c"}>] manual_axes={"a", "b", "c"} (%arg1: tensor<8x4xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg1 : tensor<8x4xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<8x4xf32>
  // CHECK-NEXT:     } : (tensor<16x16xf32>) -> tensor<32x4xf32>
  // CHECK-NEXT:     return %0 : tensor<32x4xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"b"}, {"a"}]>]>} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x16xf32>) -> tensor<8x4xf32>
  %2 = call @shmap_body_7(%1) : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"a"}, {}], replicated={"c"}>]>} : (tensor<8x4xf32>) -> tensor<32x4xf32>
  return %4 : tensor<32x4xf32>
}
// CHECK-NOT: func.func private @shmap_body_5
func.func private @shmap_body_7(%arg0: tensor<8x4xf32>) -> (tensor<8x4xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func public @shard_map_sharding_custom_call_other_uses
func.func public @shard_map_sharding_custom_call_other_uses(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  // CHECk-NEXT:     %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>}
  // CHECK:          %1 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %2 = stablehlo.add %arg1, %arg1 : tensor<4x32xf32>
  // CHECK-NEXT:       sdy.return %2 : tensor<4x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %1, %0
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = call @shmap_body_8(%1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  return %4, %0 : tensor<16x32xf32>, tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body_8(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_unused_results
func.func public @shard_map_unused_results(%arg0: tensor<16x32xf32>) -> tensor<128x32xf32> {
  // CHECK:          %[[SHARD_MAP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_1, [{}, {}], replicated={"a", "b"}>] out_shardings=[<@mesh_1, [{"b", "a"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       %[[ADD:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:       %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[ADD]]
  // CHECK-NEXT:       sdy.return %[[ADD]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<128x32xf32>
  // CHECK-NEXT:     return %[[SHARD_MAP]]
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}], replicated={"a", "b"}>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2:3 = call @shmap_body_9(%1) : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>)
  %3 = stablehlo.custom_call @Sharding(%2#1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b", "a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<128x32xf32>
  return %4 : tensor<128x32xf32>
}
// CHECK-NOT: func.func private @shmap_body_9
func.func private @shmap_body_9(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  %1 = stablehlo.multiply %0, %0 : tensor<16x32xf32>
  return %0, %0, %1 : tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_multiple_call_ops_unused_result_in_one
func.func public @shard_map_multiple_call_ops_unused_result_in_one(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>, tensor<4x128xf32>) {
  // CHECK-NEXT:     %[[SHARD_MAP_0:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:       sdy.return %[[ADD_0]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     %[[SHARD_MAP_1:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>, <@mesh_0, [{}, {"a"}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>) {
  // CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:       sdy.return %[[ADD_1]], %[[ADD_1]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<4x128xf32>)
  // CHECK-NEXT:     return %[[SHARD_MAP_0]], %[[SHARD_MAP_1]]#0, %[[SHARD_MAP_1]]#1
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2:2 = call @shmap_body_10(%1) : (tensor<4x32xf32>) -> (tensor<4x32xf32>, tensor<4x32xf32>)
  %3 = stablehlo.custom_call @Sharding(%2#0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>

  %5:2 = call @shmap_body_10(%1) : (tensor<4x32xf32>) -> (tensor<4x32xf32>, tensor<4x32xf32>)
  %6 = stablehlo.custom_call @Sharding(%5#0) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  %8 = stablehlo.custom_call @Sharding(%5#1) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %9 = stablehlo.custom_call @SPMDShardToFullShape(%8) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"a"}]>]>} : (tensor<4x32xf32>) -> tensor<4x128xf32>

  return %4, %7, %9 : tensor<16x32xf32>, tensor<16x32xf32>, tensor<4x128xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body_10(%arg0: tensor<4x32xf32>) -> (tensor<4x32xf32>, tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x32xf32>
  return %0, %0 : tensor<4x32xf32>, tensor<4x32xf32>
}

// CHECK-LABEL: func.func public @shard_map_duplicate_operand
func.func public @shard_map_duplicate_operand(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK:          %0 = sdy.manual_computation(%arg0, %arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_0, [{"a"}, {}]>, <@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x32xf32>, %arg2: tensor<4x32xf32>) {
  // CHECK-NEXT:       %1 = stablehlo.add %arg1, %arg2 : tensor<4x32xf32>
  // CHECK-NEXT:       sdy.return %1 : tensor<4x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  // CHECK-NEXT:     return %0 : tensor<16x32xf32>
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = call @shmap_body_11(%1, %1) : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<4x32xf32>) -> tensor<4x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<4x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
// CHECK-NOT: func.func private @shmap_body
func.func private @shmap_body_11(%arg0: tensor<4x32xf32>, %arg1: tensor<4x32xf32>) -> (tensor<4x32xf32>) {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
