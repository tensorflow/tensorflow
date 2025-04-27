// RUN: sdy_opt %s -xla-sdy-stablehlo-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>
sdy.mesh @mesh_2 = <["a"=4, "b"=4, "c"=4]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {mhlo.sharding = "{devices=[4,2]<=[8]}"} : tensor<8x16xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<8x16xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %arg1 {mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : tensor<16x32xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{manual}"} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %4 = stablehlo.add %1, %1 {mhlo.sharding = "{manual}"} : tensor<2x8xf32>
  // CHECK-NEXT: %5 = stablehlo.dot %4, %3 {mhlo.sharding = "{manual}"} : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %6 = "stablehlo.all_reduce"(%5)
  // CHECK:      %7 = mhlo.copy %6 {mhlo.sharding = "{manual}"} : tensor<2x32xf32>
  // CHECK-NEXT: %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %8 : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"b"}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %1 {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) {mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} : (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %4 = mhlo.copy %3 {mhlo.sharding = "{devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : tensor<8x8xf32>
  // CHECK-NEXT: %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {mhlo.sharding = "{devices=[1,1,2,4]<=[4,2]T(1,0) last_tile_dims={manual, replicated}}"} : (tensor<8x8xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %6 = mhlo.copy %5 {mhlo.sharding = "{devices=[1,1,2,4]<=[4,2]T(1,0) last_tile_dims={manual, replicated}}"} : tensor<8x4xf32>
  // CHECK-NEXT: %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {mhlo.sharding = "{devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %7 : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"b"}]>] out_shardings=[<@mesh_0, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_in_manual_computation_body
func.func @sharding_in_manual_computation_body(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,2,4]<=[16] last_tile_dim_replicate}"} : tensor<8x16xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<8x16xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %arg1 {mhlo.sharding = "{devices=[2,1,8]<=[2,2,4]T(1,0,2) last_tile_dim_replicate}"} : tensor<16x32xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %4 = stablehlo.add %1, %1 {mhlo.sharding = "{devices=[2,1,4,2]<=[4,2,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<4x8xf32>
  // CHECK-NEXT: %5 = stablehlo.dot %4, %3 {mhlo.sharding = "{devices=[2,2,4]<=[4,2,2]T(2,1,0) last_tile_dims={manual}}"} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
  // CHECK-NEXT: %6 = mhlo.copy %5 {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : tensor<4x32xf32>
  // CHECK-NEXT: %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : (tensor<4x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %7 : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_1, [{"a"}, {"b"}]>, <@mesh_1, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_1, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<4x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>} : tensor<4x8xf32>
    %2 = stablehlo.dot %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"d"}, {"c"}]>]>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    sdy.return %2 : tensor<4x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @call_op_with_no_operands_or_results
func.func @call_op_with_no_operands_or_results() {
  // CHECK-LABEL: %cst = stablehlo.constant
  // CHECK-NOT:     sdy.sharding
  // CHECK-NOT:     mhlo.sharding
  // CHECK-NEXT: %0 = stablehlo.add %cst, %cst {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<2x2xf32>
  // CHECK-NEXT: return
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<2x2xf32>
    sdy.return
  } : () -> ()
  return
}

// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}, {"c"}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}, {?}]>}) {
  // CHECK-NEXT: %[[COPY_OPERAND_LEVEL_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,1,1,8]<=[16] last_tile_dim_replicate}"} : tensor<4x8x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_LEVEL_0:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_LEVEL_0]]) {mhlo.sharding = "{devices=[1,1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<4x8x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %[[COPY_OPERAND_LEVEL_1:.*]] = mhlo.copy %[[FULL_TO_SHARD_LEVEL_0]] {mhlo.sharding = "{devices=[1,2,1,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x8x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_LEVEL_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_LEVEL_1]]) {mhlo.sharding = "{devices=[1,1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  // CHECK-NEXT: %[[COPY_OPERAND_LEVEL_2:.*]] = mhlo.copy %[[FULL_TO_SHARD_LEVEL_1]] {mhlo.sharding = "{devices=[1,1,2,4,2]<=[4,2,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x4x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_LEVEL_2:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_LEVEL_2]]) {mhlo.sharding = "{devices=[1,1,1,8,2]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<2x4x16xf32>) -> tensor<2x4x8xf32>
  // CHECK-NEXT: %[[MULT:.*]] = stablehlo.multiply %[[FULL_TO_SHARD_LEVEL_2]], %[[FULL_TO_SHARD_LEVEL_2]] {mhlo.sharding = "{devices=[1,1,1,8,2]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x4x8xf32>
  // CHECK-NEXT: %[[COPY_RESULT_LEVEL_2:.*]] = mhlo.copy %[[MULT]] {mhlo.sharding = "{devices=[1,1,1,8,2]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x4x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_LEVEL_2:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_LEVEL_2]]) {mhlo.sharding = "{devices=[1,1,2,4,2]<=[4,2,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  // CHECK-NEXT: %[[COPY_RESULT_LEVEL_1:.*]] = mhlo.copy %[[SHARD_TO_FULL_LEVEL_2]] {mhlo.sharding = "{devices=[1,1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x4x16xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_LEVEL_1:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_LEVEL_1]]) {mhlo.sharding = "{devices=[1,2,1,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x4x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %[[COPY_RESULT_LEVEL_0:.*]] = mhlo.copy %[[SHARD_TO_FULL_LEVEL_1]] {mhlo.sharding = "{devices=[1,1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x8x16xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_LEVEL_0:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_LEVEL_0]]) {mhlo.sharding = "{devices=[2,1,1,8]<=[16] last_tile_dim_replicate}"} : (tensor<2x8x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL_LEVEL_0]] : tensor<4x8x16xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8x16xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}, {}]>] out_shardings=[<@mesh_1, [{}, {"b"}, {}]>] manual_axes={"b"} (%arg2: tensor<2x4x16xf32>) {
      %2 = sdy.manual_computation(%arg2) in_shardings=[<@mesh_1, [{}, {}, {"c"}]>] out_shardings=[<@mesh_1, [{}, {}, {"c"}]>] manual_axes={"c"} (%arg3: tensor<2x4x8xf32>) {
        %3 = stablehlo.multiply %arg3, %arg3 : tensor<2x4x8xf32>
        sdy.return %3 : tensor<2x4x8xf32>
      } : (tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
      sdy.return %2 : tensor<2x4x16xf32>
    } : (tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
    sdy.return %1 : tensor<2x8x16xf32>
  } : (tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[COPY_OPERAND_OUTER:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_OUTER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_OUTER]]) {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_OPERAND_INNER:.*]] = mhlo.copy %[[FULL_TO_SHARD_OUTER]] {mhlo.sharding = "{devices=[1,2,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_INNER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_INNER]]) {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  // CHECK-NEXT: %[[MULT:.*]] = stablehlo.multiply %[[FULL_TO_SHARD_INNER]], %[[FULL_TO_SHARD_INNER]] {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MULT]], %[[MULT]] {mhlo.sharding = "{devices=[2,1,4,2]<=[4,2,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %[[ADD]], %[[ADD]] {mhlo.sharding = "{devices=[4,1,4]<=[4,4]T(1,0) last_tile_dims={manual}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[COPY_RESULT_INNER:.*]]  = mhlo.copy %[[SUB]] {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"}  : tensor<2x4xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_INNER:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_INNER]]) {mhlo.sharding = "{devices=[1,2,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x4xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[SHARD_TO_FULL_INNER]], %[[SHARD_TO_FULL_INNER]] {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_RESULT_OUTER:.*]] = mhlo.copy %[[ADD]] {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_OUTER:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_OUTER]]) {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL_OUTER]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      %3 = stablehlo.add %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>} : tensor<2x4xf32>
      %4 = stablehlo.subtract %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c", "d"}, {}]>]>} : tensor<2x4xf32>
      sdy.return %4 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %5 = stablehlo.add %1, %1 : tensor<2x8xf32>
    sdy.return %5 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multiple_manual_computation_uses
func.func @multiple_manual_computation_uses(%arg0: tensor<2x4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {"a"}]>}, %arg1: tensor<32x16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {"a"}]>}) -> (tensor<131x4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{?}, {?}, {"a"}]>}) {
  // CHECK-NEXT: %[[COPY_OPERAND_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : tensor<2x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_0:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_0]]) {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<2x4x8xi32>) -> tensor<2x4x2xi32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @sdy_testonly(%[[FULL_TO_SHARD_0]])  {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[COPY_RESULT_0:.*]] = mhlo.copy %[[CUSTOM_CALL]] {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_0:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_0]]) {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : (tensor<3x4x2xi32>) -> tensor<3x4x8xi32>
  // CHECK-NEXT: %[[COPY_OPERAND_1:.*]] = mhlo.copy %arg1 {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : tensor<32x16x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_1]])  {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<32x16x8xi32>) -> tensor<32x16x2xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[FULL_TO_SHARD_1]] {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32>
  // CHECK-NEXT: %[[COPY_RESULT_1:.*]] = mhlo.copy %[[RESHAPE]] {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : tensor<128x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_1:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_1]]) {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : (tensor<128x4x2xi32>) -> tensor<128x4x8xi32>
  // CHECK-NEXT: %[[COPY_OPERAND_2:.*]] = mhlo.copy %[[SHARD_TO_FULL_0]] {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : tensor<3x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_2:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_2]]) {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<3x4x8xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[COPY_OPERAND_3:.*]] = mhlo.copy %[[SHARD_TO_FULL_1]] {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : tensor<128x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_3:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_3]]) {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<128x4x8xi32>) -> tensor<128x4x2xi32>
  // CHECK-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[FULL_TO_SHARD_3]], %[[FULL_TO_SHARD_2]], dim = 0 {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : (tensor<128x4x2xi32>, tensor<3x4x2xi32>) -> tensor<131x4x2xi32>
  // CHECK-NEXT: %[[COPY_RESULT_2:.*]] = mhlo.copy %[[CONCAT]] {mhlo.sharding = "{devices=[1,1,1,4,2]<=[8] last_tile_dims={manual, replicated}}"} : tensor<131x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_2:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_2]]) {mhlo.sharding = "{devices=[1,1,4,2]<=[8] last_tile_dim_replicate}"} : (tensor<131x4x2xi32>) -> tensor<131x4x8xi32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL_2]] : tensor<131x4x8xi32>
  %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<2x4x2xi32>) {
    %4 = stablehlo.custom_call @sdy_testonly(%arg2) : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
    sdy.return %4 : tensor<3x4x2xi32>
  } : (tensor<2x4x8xi32>) -> tensor<3x4x8xi32>
  %2 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<32x16x2xi32>) {
    %4 = stablehlo.reshape %arg2 : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32>
    sdy.return %4 : tensor<128x4x2xi32>
  } : (tensor<32x16x8xi32>) -> tensor<128x4x8xi32>
  %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>, <@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<3x4x2xi32>, %arg3: tensor<128x4x2xi32>) {
    %4 = stablehlo.concatenate %arg3, %arg2, dim = 0 : (tensor<128x4x2xi32>, tensor<3x4x2xi32>) -> tensor<131x4x2xi32>
    sdy.return %4 : tensor<131x4x2xi32>
  } : (tensor<3x4x8xi32>, tensor<128x4x8xi32>) -> tensor<131x4x8xi32>
  return %3 : tensor<131x4x8xi32>
}

// CHECK-LABEL: func @remove_auto_axes_to_avoid_padding
func.func @remove_auto_axes_to_avoid_padding(%arg0: tensor<4xf32>, %arg1: tensor<12xf32>, %arg2: tensor<24xf32>, %arg3: tensor<48xf32>, %arg4: tensor<96xf32>, %arg5: tensor<192xf32>) -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {mhlo.sharding = "{devices=[4,16]<=[64] last_tile_dim_replicate}"} : tensor<4xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{devices=[1,4,16]<=[64] last_tile_dims={manual, replicated}}"} : (tensor<4xf32>) -> tensor<1xf32>
  // CHECK-NEXT: %2 = mhlo.copy %arg1 {mhlo.sharding = "{devices=[4,16]<=[64] last_tile_dim_replicate}"} : tensor<12xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{devices=[1,4,16]<=[64] last_tile_dims={manual, replicated}}"} : (tensor<12xf32>) -> tensor<3xf32>
  // CHECK-NEXT: %4 = mhlo.copy %arg2 {mhlo.sharding = "{devices=[8,8]<=[64] last_tile_dim_replicate}"} : tensor<24xf32>
  // CHECK-NEXT: %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {mhlo.sharding = "{devices=[2,4,8]<=[4,2,8]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<24xf32>) -> tensor<6xf32>
  // CHECK-NEXT: %6 = mhlo.copy %arg3 {mhlo.sharding = "{devices=[16,4]<=[64] last_tile_dim_replicate}"} : tensor<48xf32>
  // CHECK-NEXT: %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {mhlo.sharding = "{devices=[4,4,4]<=[4,4,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<48xf32>) -> tensor<12xf32>
  // CHECK-NEXT: %8 = mhlo.copy %arg4 {mhlo.sharding = "{devices=[32,2]<=[64] last_tile_dim_replicate}"} : tensor<96xf32>
  // CHECK-NEXT: %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {mhlo.sharding = "{devices=[8,4,2]<=[4,8,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<96xf32>) -> tensor<24xf32>
  // CHECK-NEXT: %10 = mhlo.copy %arg5 {mhlo.sharding = "{devices=[64]<=[64]}"} : tensor<192xf32>
  // CHECK-NEXT: %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {mhlo.sharding = "{devices=[16,4]<=[4,16]T(1,0) last_tile_dims={manual}}"} : (tensor<192xf32>) -> tensor<48xf32>
  // CHECK-NEXT: %12 = mhlo.copy %1 {mhlo.sharding = "{devices=[1,4,16]<=[64] last_tile_dims={manual, replicated}}"} : tensor<1xf32>
  // CHECK-NEXT: %13 = stablehlo.custom_call @SPMDShardToFullShape(%12) {mhlo.sharding = "{devices=[4,16]<=[64] last_tile_dim_replicate}"} : (tensor<1xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %14 = mhlo.copy %3 {mhlo.sharding = "{devices=[1,4,16]<=[64] last_tile_dims={manual, replicated}}"} : tensor<3xf32>
  // CHECK-NEXT: %15 = stablehlo.custom_call @SPMDShardToFullShape(%14) {mhlo.sharding = "{devices=[4,16]<=[64] last_tile_dim_replicate}"} : (tensor<3xf32>) -> tensor<12xf32>
  // CHECK-NEXT: %16 = mhlo.copy %5 {mhlo.sharding = "{devices=[2,4,8]<=[4,2,8]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<6xf32>
  // CHECK-NEXT: %17 = stablehlo.custom_call @SPMDShardToFullShape(%16) {mhlo.sharding = "{devices=[8,8]<=[64] last_tile_dim_replicate}"} : (tensor<6xf32>) -> tensor<24xf32>
  // CHECK-NEXT: %18 = mhlo.copy %7 {mhlo.sharding = "{devices=[4,4,4]<=[4,4,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<12xf32>
  // CHECK-NEXT: %19 = stablehlo.custom_call @SPMDShardToFullShape(%18) {mhlo.sharding = "{devices=[16,4]<=[64] last_tile_dim_replicate}"} : (tensor<12xf32>) -> tensor<48xf32>
  // CHECK-NEXT: %20 = mhlo.copy %9 {mhlo.sharding = "{devices=[8,4,2]<=[4,8,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<24xf32>
  // CHECK-NEXT: %21 = stablehlo.custom_call @SPMDShardToFullShape(%20) {mhlo.sharding = "{devices=[32,2]<=[64] last_tile_dim_replicate}"} : (tensor<24xf32>) -> tensor<96xf32>
  // CHECK-NEXT: %22 = mhlo.copy %11 {mhlo.sharding = "{devices=[16,4]<=[4,16]T(1,0) last_tile_dims={manual}}"} : tensor<48xf32>
  // CHECK-NEXT: %23 = stablehlo.custom_call @SPMDShardToFullShape(%22) {mhlo.sharding = "{devices=[64]<=[64]}"} : (tensor<48xf32>) -> tensor<192xf32>
  // CHECK-NEXT: return %13, %15, %17, %19, %21, %23 : tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>
  %0:6 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
    in_shardings=[<@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>]
    out_shardings=[<@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>, <@mesh_2, [{"a", "b", "c"}]>]
    manual_axes={"a"} (%arg6: tensor<1xf32>, %arg7: tensor<3xf32>, %arg8: tensor<6xf32>, %arg9: tensor<12xf32>, %arg10: tensor<24xf32>, %arg11: tensor<48xf32>) {
    sdy.return %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : tensor<1xf32>, tensor<3xf32>, tensor<6xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>
  } : (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>
}
