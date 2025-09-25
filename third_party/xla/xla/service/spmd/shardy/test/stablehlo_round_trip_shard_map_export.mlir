// RUN: sdy_opt %s -xla-sdy-stablehlo-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>
sdy.mesh @mesh_2 = <["a"=4, "b"=4, "c"=4]>
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {"b"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<8x16xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}, {}], replicated={"a"}>]>} : tensor<16x32xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body(%[[FULL_TO_SHARD]], %[[FULL_TO_SHARD_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x32xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}], replicated={"b"}>]>} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<8x32xf32>
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
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_0(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[SHARD_TO_FULL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"b"}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<8x8xf32>) -> tensor<8x4xf32>
   // CHECK-NEXT: %[[CALL_1:.*]] = call @xla.sdy.inlinable_manual_computation_body_1(%[[FULL_TO_SHARD_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[COPY_3:.*]] = mhlo.copy %[[CALL_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : tensor<8x4xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_1:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_3]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"b"}]>]>} : (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL_1]] : tensor<8x8xf32>
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
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {"b"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<8x16xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b"}, {}], replicated={"a"}>]>} : tensor<16x32xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_2(%[[FULL_TO_SHARD]], %[[FULL_TO_SHARD_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}], replicated={"b"}>]>} : (tensor<4x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<8x32xf32>
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
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}, {}]>]>} : tensor<4x8x16xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_5(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8x16xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}, {}]>]>} : (tensor<2x8x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<4x8x16xf32>
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
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<4x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_7(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<4x8xf32>
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
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<2x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x8xi32>) -> tensor<2x4x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_8(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<3x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<3x4x2xi32>) -> tensor<3x4x8xi32>
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<32x16x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_2:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32x16x8xi32>) -> tensor<32x16x2xi32>
  // CHECK-NEXT: %[[CALL_1:.*]] = call @xla.sdy.inlinable_manual_computation_body_9(%[[FULL_TO_SHARD_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32
  // CHECK-NEXT: %[[COPY_3:.*]] = mhlo.copy %[[CALL_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<128x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_1:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_3]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<128x4x2xi32>) -> tensor<128x4x8xi32>
  // CHECK-NEXT: %[[COPY_4:.*]] = mhlo.copy %[[SHARD_TO_FULL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<3x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_3:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_4]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<3x4x8xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %[[COPY_5:.*]] = mhlo.copy %[[SHARD_TO_FULL_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<128x4x8xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD_4:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_5]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<128x4x8xi32>) -> tensor<128x4x2xi32>
  // CHECK-NEXT: %[[CALL_2:.*]] = call @xla.sdy.inlinable_manual_computation_body_10(%[[FULL_TO_SHARD_3]], %[[FULL_TO_SHARD_4]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<3x4x2xi32>, tensor<128x4x2xi32>) -> tensor<131x4x2xi32>
  // CHECK-NEXT: %[[COPY_6:.*]] = mhlo.copy %[[CALL_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<131x4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_2:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_6]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<131x4x2xi32>) -> tensor<131x4x8xi32>
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

// CHECK-LABEL: func @named_computation_in_manual_computation
func.func @named_computation_in_manual_computation(%arg0: tensor<32xi32>) -> (tensor<32xi32>, tensor<32xi32>, tensor<16xi32>) {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : tensor<32xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32xi32>) -> tensor<8xi32>
  // CHECK-NEXT: %[[CALL:.*]]:3 = call @xla.sdy.inlinable_manual_computation_body_11(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>, tensor<4xi32>)
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[CALL]]#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<8xi32>) -> tensor<32xi32>
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[CALL]]#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_1:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<8xi32>) -> tensor<32xi32>
  // CHECK-NEXT: %[[COPY_3:.*]] = mhlo.copy %[[CALL]]#2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<4xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL_2:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_3]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<4xi32>) -> tensor<16xi32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]], %[[SHARD_TO_FULL_1]], %[[SHARD_TO_FULL_2]] : tensor<32xi32>, tensor<32xi32>, tensor<16xi32>
  %0:3 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a", "b"}]>] out_shardings=[<@mesh_0, [{"a", "b"}]>, <@mesh_0, [{"a", "b"}]>, <@mesh_0, [{"a", "b"}]>] manual_axes={"a"} (%arg1: tensor<8xi32>) {
    %1:2 = sdy.named_computation<"foo">(%arg1) out_shardings=[<@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>] (%arg2: tensor<8xi32>) {
      %2 = stablehlo.multiply %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : tensor<8xi32>
      %3 = stablehlo.negate %arg2 : tensor<8xi32>
      sdy.return %2, %3 : tensor<8xi32>, tensor<8xi32>
    } : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
    %4 = sdy.named_computation<"no_input_named_computation">() () {
      %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
      %5 = stablehlo.negate %c : tensor<4xi32>
      sdy.return %5 : tensor<4xi32>
    } : () -> (tensor<4xi32>)
    sdy.return %1#0, %1#1, %4 : tensor<8xi32>, tensor<8xi32>, tensor<4xi32>
  } : (tensor<32xi32>) -> (tensor<32xi32>, tensor<32xi32>, tensor<16xi32>)
  return %0#0, %0#1, %0#2 : tensor<32xi32>, tensor<32xi32>, tensor<16xi32>
}

// CHECK-LABEL: func @manual_computation_with_tokens
func.func @manual_computation_with_tokens(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh_0, []>},
    %arg1: tensor<2xi64> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>}
) -> (!stablehlo.token, tensor<2xi64>) {
  // CHECK-NEXT: %[[COPY_OPERAND:.*]] = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : tensor<2xi64>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @xla.sdy.inlinable_manual_computation_body_12(%arg0, %[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>, <@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (!stablehlo.token, tensor<1xi64>) -> (!stablehlo.token, tensor<1xi64>)
  // CHECK-NEXT: %[[COPY_RESULT:.*]] = mhlo.copy %[[CALL]]#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<1xi64>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : (tensor<1xi64>) -> tensor<2xi64>
  // CHECK-NEXT: return %[[CALL]]#0, %[[SHARD_TO_FULL]] : !stablehlo.token, tensor<2xi64>
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
      out_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
      manual_axes={"a", "b"} (%arg2: !stablehlo.token, %arg3: tensor<1xi64>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg2) : (!stablehlo.token) -> (!stablehlo.token)
    stablehlo.custom_call @sdy_testonly(%1) {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (!stablehlo.token) -> ()
    sdy.return %1, %arg3 : !stablehlo.token, tensor<1xi64>
  } : (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<2xi64>
}


// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>},
// CHECK-SAME{LITERAL}:     %arg1: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}) {
// CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x8xf32>
// CHECK-NEXT:            %[[DOT:.*]] = stablehlo.dot %[[ADD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:            %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%[[DOT]]) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<{{\[\[}}0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
// CHECK-NEXT:            ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
// CHECK-NEXT:              %[[ADD_1:.*]] = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<f32>
// CHECK-NEXT:              stablehlo.return %[[ADD_1]] {xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<f32>
// CHECK-NEXT:            }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:            return %[[ALL_REDUCE]] : tensor<2x32xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_0
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            return %arg0 : tensor<2x8xf32>
// CHECK-NEXT:          }


// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_1
// CHECK-SAME{LITERAL}:     %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>}) {
// CHECK-NEXT:            return %arg0 : tensor<8x4xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_2
// CHECK-SAME{LITERAL}:     %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>},
// CHECK-SAME{LITERAL}:     %arg1: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}) {
// CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<4x8xf32>
// CHECK-NEXT:            %[[DOT:.*]] = stablehlo.dot %[[ADD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"d"}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT:            return %[[DOT]] : tensor<4x32xf32>
// CHECK-NEXT:          }


// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_3
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>}) {
// CHECK-NEXT:            %[[MULT:.*]] = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : tensor<2x4x8xf32>
// CHECK-NEXT:            return %[[MULT]] : tensor<2x4x8xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_4
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}) {
// CHECK-NEXT:            %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4x16xf32>
// CHECK-NEXT:            %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : (tensor<2x4x16xf32>) -> tensor<2x4x8xf32>
// CHECK-NEXT:            %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_3(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK-NEXT:            %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : tensor<2x4x8xf32>
// CHECK-NEXT:            %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
// CHECK-NEXT:            return %[[SHARD_TO_FULL]] : tensor<2x4x16xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_5
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8x16xf32>
// CHECK-NEXT:            %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
// CHECK-NEXT:            %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_4(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
// CHECK-NEXT:            %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4x16xf32>
// CHECK-NEXT:            %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x16xf32>) -> tensor<2x8x16xf32>
// CHECK-NEXT:            return %[[SHARD_TO_FULL]] : tensor<2x8x16xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_6
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}) {
// CHECK-NEXT:            %[[MULT:.*]] = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
// CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %[[MULT]], %[[MULT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
// CHECK-NEXT:            %[[SUB:.*]] = stablehlo.subtract %[[ADD]], %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c", "d"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
// CHECK-NEXT:            return %[[SUB]] : tensor<2x4xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_7
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[COPY:.*]] = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
// CHECK-NEXT:            %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:            %[[CALL:.*]] = call @xla.sdy.inlinable_manual_computation_body_6(%[[FULL_TO_SHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:            %[[COPY_1:.*]] = mhlo.copy %[[CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
// CHECK-NEXT:            %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:            %[[ADD:.*]] = stablehlo.add %[[SHARD_TO_FULL]], %[[SHARD_TO_FULL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
// CHECK-NEXT:            return %[[ADD]] : tensor<2x8xf32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_8
// CHECK-SAME{LITERAL}:     %arg0: tensor<2x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<3x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
// CHECK-NEXT:            return %[[CUSTOM_CALL]] : tensor<3x4x2xi32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_9
// CHECK-SAME{LITERAL}:     %arg0: tensor<32x16x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<128x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32>
// CHECK-NEXT:            return %[[RESHAPE]] : tensor<128x4x2xi32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_10
// CHECK-SAME{LITERAL}:     %arg0: tensor<3x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>},
// CHECK-SAME{LITERAL}:     %arg1: tensor<128x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) -> (tensor<131x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[CONCAT:.*]] = stablehlo.concatenate %arg1, %arg0, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<128x4x2xi32>, tensor<3x4x2xi32>) -> tensor<131x4x2xi32>
// CHECK-NEXT:            return %[[CONCAT]] : tensor<131x4x2xi32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_11
// CHECK-SAME{LITERAL}:     %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>})
// CHECK-SAME{LITERAL}:     -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}, tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>},
// CHECK-SAME{LITERAL}:         tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>}) {
// CHECK-NEXT:            %[[MC:.*]]:2 = sdy.named_computation<"foo">(%arg0) in_shardings=[<@mesh_0, [{}]>] out_shardings=[<@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>] (%arg1: tensor<8xi32>) {
// CHECK-NEXT:              %[[MULT:.*]] = stablehlo.multiply %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT:              %[[NEGATE:.*]] = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT:              sdy.return {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} %[[MULT]], %[[NEGATE]] : tensor<8xi32>, tensor<8xi32>
// CHECK-NEXT:            } {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
// CHECK-NEXT:            %[[NC:.*]] = sdy.named_computation<"no_input_named_computation">() in_shardings=[] out_shardings=[<@mesh_0, [{}]>] () {
// CHECK-NEXT:              %[[CONST:.*]] = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-NEXT:              %[[NEGATE_1:.*]] = stablehlo.negate %[[CONST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<4xi32>
// CHECK-NEXT:              sdy.return {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} %[[NEGATE_1]] : tensor<4xi32>
// CHECK-NEXT:            } {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : () -> tensor<4xi32>
// CHECK-NEXT:            return %[[MC]]#0, %[[MC]]#1, %[[NC]] : tensor<8xi32>, tensor<8xi32>, tensor<4xi32>
// CHECK-NEXT:          }

// CHECK-LABEL:         func @xla.sdy.inlinable_manual_computation_body_12
// CHECK-SAME{LITERAL}:     %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh_0, []>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>},
// CHECK-SAME{LITERAL}:     %arg1: tensor<1xi64> {sdy.sharding = #sdy.sharding<@mesh_0, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>})
// CHECK-SAME{LITERAL}:     -> (!stablehlo.token {sdy.sharding = #sdy.sharding<@mesh_0, []>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}, tensor<1xi64> {sdy.sharding = #sdy.sharding<@mesh_0, [{}]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>}) {
// CHECK-NEXT:            %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (!stablehlo.token) -> !stablehlo.token
// CHECK-NEXT:            stablehlo.custom_call @sdy_testonly(%[[CUSTOM_CALL]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (!stablehlo.token) -> ()
// CHECK-NEXT:            return %[[CUSTOM_CALL]], %arg1 : !stablehlo.token, tensor<1xi64>
// CHECK-NEXT:          }
