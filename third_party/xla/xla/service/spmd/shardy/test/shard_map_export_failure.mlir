// RUN: sdy_opt %s -xla-sdy-shard-map-export -split-input-file -verify-diagnostics

module @main {
  sdy.mesh @mesh_0 = <"axis_0"=4, "axis_1"=2, "axis_2"=2>
  func.func @main(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0", ?}, {"axis_1", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_1", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0"}, {}]>}) {
    // expected-error @+1 {{failed to legalize operation 'sdy.manual_computation' that was explicitly marked illegal}}
    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"axis_0"}, {"axis_1"}]>, <@mesh_0, [{"axis_1"}, {}], replicated={"axis_0"}>] out_shardings=[<@mesh_0, [{"axis_0"}, {}], replicated={"axis_1"}>] manual_axes={"axis_0", "axis_1"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
      // expected-error @+1 {{Operation in ManualComputationOp has a sdy.sharding attribute.}}
      %1 = mhlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_2"}, {}]>]>} : tensor<2x8xf32>
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
}
