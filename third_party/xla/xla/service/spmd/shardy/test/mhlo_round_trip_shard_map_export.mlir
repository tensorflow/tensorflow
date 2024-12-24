// RUN: sdy_opt %s -xla-sdy-mhlo-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

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
func.func @nested_shmaps(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[COPY_OPERAND_OUTER:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_OUTER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_OUTER]]) {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_OPERAND_INNER:.*]] = mhlo.copy %[[FULL_TO_SHARD_OUTER]] {mhlo.sharding = "{devices=[1,2,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_INNER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_INNER]]) {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  // CHECK-NEXT: %[[MULT:.*]] = stablehlo.multiply %[[FULL_TO_SHARD_INNER]], %[[FULL_TO_SHARD_INNER]] {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[COPY_RESULT_INNER:.*]]  = mhlo.copy %[[MULT]] {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_INNER:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_INNER]]) {mhlo.sharding = "{devices=[1,2,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x4xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_RESULT_OUTER:.*]] = mhlo.copy %[[SHARD_TO_FULL_INNER]] {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_OUTER:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT_OUTER]]) {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL_OUTER]] : tensor<4x8xf32>
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
  // CHECK-NEXT: %[[COPY_OPERAND_OUTER:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_OUTER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_OUTER]]) {mhlo.sharding = "{devices=[1,1,2,8]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_OPERAND_INNER:.*]] = mhlo.copy %[[FULL_TO_SHARD_OUTER]] {mhlo.sharding = "{devices=[1,2,2,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_INNER:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND_INNER]]) {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  // CHECK-NEXT: %[[MULT:.*]] = stablehlo.multiply %[[FULL_TO_SHARD_INNER]], %[[FULL_TO_SHARD_INNER]] {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MULT]], %[[MULT]] {mhlo.sharding = "{devices=[2,1,4,2]<=[2,2,2,2]T(2,1,0,3) last_tile_dims={manual, replicated}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %[[ADD]], %[[ADD]] {mhlo.sharding = "{devices=[4,1,4]<=[2,2,4]T(2,1,0) last_tile_dims={manual}}"} : tensor<2x4xf32>
  // CHECK-NEXT: %[[COPY_RESULT_INNER:.*]]  = mhlo.copy %[[SUB]] {mhlo.sharding = "{devices=[1,1,4,4]<=[2,2,4]T(1,0,2) last_tile_dims={manual, replicated}}"}  : tensor<2x4xf32>
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
