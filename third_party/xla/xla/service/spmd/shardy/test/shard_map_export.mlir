// RUN: sdy_opt %s -xla-sdy-shard-map-export 2>&1 | FileCheck %s

module {
  sdy.mesh @mesh_0 = <"axis_0"=4, "axis_1"=2>

  // CHECK-LABEL: func @single_manual_comp
  func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0", ?}, {"axis_1", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_1", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0"}, {}]>}) {
    // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_0"}, {"axis_1"}]>]>} : tensor<8x16xf32>
    // CHECK-NEXT: %1 = mhlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<8x16xf32>) -> tensor<2x8xf32>
    // CHECK-NEXT: %2 = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_1"}, {}], replicated={"axis_0"}>]>} : tensor<16x32xf32>
    // CHECK-NEXT: %3 = mhlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{manual}"} : (tensor<16x32xf32>) -> tensor<8x32xf32>
    // CHECK-NEXT: %4 = mhlo.add %1, %1 {mhlo.sharding = "{manual}"} : tensor<2x8xf32>
    // CHECK-NEXT: %5 = "mhlo.dot"(%4, %3) <{precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}> {mhlo.sharding = "{manual}"} : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    // CHECK-NEXT: %6 = "mhlo.all_reduce"(%5)
    // CHECK:      %7 = mhlo.copy %6 {mhlo.sharding = "{manual}"} : tensor<2x32xf32>
    // CHECK-NEXT: %8 = mhlo.custom_call @SPMDShardToFullShape(%7) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_0"}, {}], replicated={"axis_1"}>]>} : (tensor<2x32xf32>) -> tensor<8x32xf32>
    // CHECK-NEXT: return %8 : tensor<8x32xf32>
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"axis_0"}, {"axis_1"}]>, <@mesh_0, [{"axis_1"}, {}], replicated={"axis_0"}>] out_shardings=[<@mesh_0, [{"axis_0"}, {}], replicated={"axis_1"}>] manual_axes={"axis_0", "axis_1"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
      %1 = mhlo.add %arg2, %arg2 : tensor<2x8xf32>
      %2 = "mhlo.dot"(%1, %arg3) <{precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}> : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
      %3 = "mhlo.all_reduce"(%2) <{channel_handle = #mhlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
        %4 = mhlo.add %arg4, %arg5 : tensor<f32>
        mhlo.return %4 : tensor<f32>
      }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
      sdy.return %3 : tensor<2x32xf32>
    } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
    return %0 : tensor<8x32xf32>
  }

  // CHECK-LABEL: func @manual_comp_using_another
  func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0"}, {}]>})
      -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_1"}]>}) {
    // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_0"}, {}]>]>} : tensor<8x8xf32>
    // CHECK-NEXT: %1 = mhlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<8x8xf32>) -> tensor<2x8xf32>
    // CHECK-NEXT: %2 = mhlo.copy %1 {mhlo.sharding = "{manual}"} : tensor<2x8xf32>
    // CHECK-NEXT: %3 = mhlo.custom_call @SPMDShardToFullShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_0"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<8x8xf32>
    // CHECK-NEXT: %4 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"axis_1"}]>]>} : tensor<8x8xf32>
    // CHECK-NEXT: %5 = mhlo.custom_call @SPMDFullToShardShape(%4) {mhlo.sharding = "{manual}"} : (tensor<8x8xf32>) -> tensor<8x4xf32>
    // CHECK-NEXT: %6 = mhlo.copy %5 {mhlo.sharding = "{manual}"} : tensor<8x4xf32>
    // CHECK-NEXT: %7 = mhlo.custom_call @SPMDShardToFullShape(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"axis_1"}]>]>} : (tensor<8x4xf32>) -> tensor<8x8xf32>
    // CHECK-NEXT: return %7 : tensor<8x8xf32>
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"axis_0"}, {}]>] out_shardings=[<@mesh_0, [{"axis_0"}, {}]>] manual_axes={"axis_0"} (%arg1: tensor<2x8xf32>) {
      sdy.return %arg1 : tensor<2x8xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>

    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"axis_1"}]>] out_shardings=[<@mesh_0, [{}, {"axis_1"}]>] manual_axes={"axis_1"} (%arg1: tensor<8x4xf32>) {
      sdy.return %arg1 : tensor<8x4xf32>
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
  }
}
