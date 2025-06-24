// RUN: sdy_opt %s -xla-sdy-stablehlo-export-pipeline 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["axis_0"=2, "axis_1"=4, "axis_2"=4]>
sdy.mesh @mesh_1 = <["axis_0"=16]>
sdy.mesh @mesh_2 = <["x"=8, "y"=4]>
sdy.mesh @mesh_3 = <["a"=2, "b"=2, "c"=2, "d"=2]>
sdy.mesh @mesh_4 = <["axis_0"=2, "axis_1"=2, "axis_2"=2], device_ids=[0,2,4,6,1,3,5,7]>
sdy.mesh @mesh_5 = <["i"=2, "j"=2]>
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
sdy.mesh @maximal_mesh_1 = <[], device_ids=[1]>
sdy.mesh @empty_mesh_0 = <[]>
sdy.mesh @empty_mesh_1 = <[]>

// CHECK-NOT: sdy.mesh

// CHECK-LABEL: func @non_trivial_common_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,16]<=[32] last_tile_dim_replicate}"},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,4,2]<=[2,16]T(1,0) last_tile_dim_replicate}"})
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @non_trivial_common_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                                   %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_0"}]>},
                                   %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_1"}, {"axis_2"}]>}) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @multiple_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,8,4]<=[2,4,4]T(0,2,1) last_tile_dim_replicate}"},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[1,4,8]<=[2,4,4]T(1,0,2) last_tile_dim_replicate}"})
// CHECK-SAME:  -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
func.func @multiple_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                              %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_0", "axis_2"}]>},
                              %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"axis_1"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"axis_0", "axis_1"}, {"axis_2"}]>}) {
// CHECK-NEXT: stablehlo.add
// CHECK-SAME{LITERAL}: {mhlo.sharding = "{devices=[8,1,4]<=[2,4,4]T(1,0,2) last_tile_dim_replicate}"}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"axis_1","axis_0"}, {}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_axis(
// CHECK-SAME:      %arg0: tensor<32x8xf32> {mhlo.sharding = "{devices=[16,1]<=[16]}"},
// CHECK-SAME:      %arg1: tensor<8x16xf32>)
// CHECK-SAME:  -> tensor<32x16xf32> {
func.func @single_axis(%arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"axis_0"}, {}]>},
                       %arg1: tensor<8x16xf32>) -> tensor<32x16xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<32x8xf32>, tensor<8x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// CHECK-LABEL: func @multi_result_op
func.func @multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: stablehlo.reduce
// CHECK-SAME{LITERAL}: {mhlo.sharding = "{{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}, {devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}}"}
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{}, {"y"}]>, <@mesh_2, [{"y"}, {}]>]>} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @fully_replicated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{replicated}"},
// CHECK-SAME:      %arg2: tensor<8x16xf32>)
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @fully_replicated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"y"}, {}]>},
                            %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>},
                            %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @split_axes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,2,4]<=[2,2,2,4]T(3,1,0,2) last_tile_dim_replicate}"},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {mhlo.sharding = "{devices=[2,4,4]<=[32] last_tile_dim_replicate}"})
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @split_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"y"}, {"x":(2)2}]>},
                      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x":(1)2}, {"x":(2)4}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: stablehlo.dot
// CHECK-SAME{LITERAL}: {mhlo.sharding = "{devices=[4,1,8]<=[2,2,2,4]T(0,2,1,3) last_tile_dim_replicate}"}
  %1 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x":(1)2, "x":(4)2}, {}]>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @split_constants
func.func @split_constants() -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = stablehlo.constant {mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = stablehlo.constant {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} dense<1.000000e+00>
  // CHECK-NEXT: return %[[CONST_0]], %[[CONST_1]]
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} dense<1.000000e+00> : tensor<8x8xf32>
  %1 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"y"}, {}]>]>} dense<1.000000e+00> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard
func.func @reshard(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,4,4]<=[2,4,4]T(0,2,1) last_tile_dim_replicate}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.reshard %arg0 <@mesh_0, [{"axis_0"}, {"axis_2"}], replicated={"axis_1"}> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_gather
func.func @all_gather(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"x"}, {"y"}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_slice
func.func @all_slice(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{}, {"y"}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[8,4]<=[32]}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_to_all
func.func @all_to_all(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"y"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.all_to_all [{"y"}: 0->1] %arg0 out_sharding=<@mesh_2, [{}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @collective_permute
func.func @collective_permute(%arg0: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"x", "y"}, {}]>}) -> tensor<32x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[32,1]<=[8,4]T(1,0)}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2, [{"y", "x"}, {}]> : tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}

// CHECK-LABEL: func @all_reduce
func.func @all_reduce(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[8,4]<=[32]}"}
  // CHECK-NEXT: return %[[COPY]]
  %0 = sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh_2, [{"x"}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_of_collectives
func.func @chain_of_collectives(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"y"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"}
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[COPY_0]] {mhlo.sharding = "{devices=[8,4]<=[32]}"}
  // CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[COPY_1]] {mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"}
  // CHECK-NEXT: return %[[COPY_2]]
  %0 = sdy.all_to_all [{"y"}: 0->1] %arg0 out_sharding=<@mesh_2, [{}, {"y"}]> : tensor<8x8xf32>
  %1 = sdy.all_slice [{"x"}, {}] %0 out_sharding=<@mesh_2, [{"x"}, {"y"}]> : tensor<8x8xf32>
  %2 = sdy.all_gather [{}, {"y"}] %1 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_in_manual_computation_body(
// CHECK-SAME:      %arg0: tensor<8x16xf32> {mhlo.sharding = "{devices=[2,2,4]<=[16] last_tile_dim_replicate}"},
// CHECK-SAME:      %arg1: tensor<16x32xf32> {mhlo.sharding = "{devices=[2,1,8]<=[2,2,4]T(1,0,2) last_tile_dim_replicate}"})
// CHECK-SAME:  -> (tensor<8x32xf32> {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"}) {
func.func @sharding_in_manual_computation_body(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_3, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_3, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_3, [{"a"}, {}]>}) {
// CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,2,4]<=[2,2,4]T(1,0,2) last_tile_dim_replicate}"} : tensor<8x16xf32>
// CHECK-NEXT: %[[FULL_TO_SHARD_0:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_0]]) {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<8x16xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %arg1 {mhlo.sharding = "{devices=[2,1,8]<=[2,2,4]T(1,0,2) last_tile_dim_replicate}"} : tensor<16x32xf32>
// CHECK-NEXT: %[[FULL_TO_SHARD_1:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_1]]) {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : (tensor<16x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: %[[RESHARD:.*]] = mhlo.copy %[[FULL_TO_SHARD_0]] {mhlo.sharding = "{devices=[1,2,4,2]<=[8,2]T(1,0) last_tile_dims={manual, replicated}}"} : tensor<4x8xf32>
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD]], %[[RESHARD]] {mhlo.sharding = "{devices=[2,1,4,2]<=[4,2,2]T(1,0,2) last_tile_dims={manual, replicated}}"} : tensor<4x8xf32>
// CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[ADD]], %[[FULL_TO_SHARD_1]] {mhlo.sharding = "{devices=[2,2,4]<=[4,4]T(1,0) last_tile_dims={manual}}"} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
// CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %[[DOT]] {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : tensor<4x32xf32>
// CHECK-NEXT: %[[COPY_2:.*]] = mhlo.copy %[[SINE]] {mhlo.sharding = "{devices=[1,1,4,4]<=[16] last_tile_dims={manual, replicated}}"} : tensor<4x32xf32>
// CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_2]]) {mhlo.sharding = "{devices=[2,1,8]<=[16] last_tile_dim_replicate}"} : (tensor<4x32xf32>) -> tensor<8x32xf32>
// CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_3, [{"b"}, {"a"}]>, <@mesh_3, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_3, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<4x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = sdy.reshard %arg2 <@mesh_3, [{}, {"d"}]> : tensor<4x8xf32>
    %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3, [{"c"}, {}]>]>} : tensor<4x8xf32>
    %3 = stablehlo.dot %2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_3, [{"c"}, {"d"}]>]>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    %4 = stablehlo.sine %3 : tensor<4x32xf32>
    sdy.return %4 : tensor<4x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @mesh_with_device_id_should_be_converted_to_maximal_sharding(%arg0: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"}, %arg1: tensor<8x8xf32>)
func.func @mesh_with_device_id_should_be_converted_to_maximal_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
    %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
    // CHECK: %[[ADD_WITH_SHARDING:.*]] = stablehlo.add %[[ADD]], %[[ADD]] {mhlo.sharding = "{maximal device=1}"}
    %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_1, []>]>} : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @mesh_empty_should_be_converted_to_replicated_sharding(%arg0: tensor<8x8xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<8x8xf32>)
func.func @mesh_empty_should_be_converted_to_replicated_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh_0, [{}, {}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
    %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
    // CHECK: %[[ADD_WITH_SHARDING:.*]] = stablehlo.add %[[ADD]], %[[ADD]] {mhlo.sharding = "{replicated}"}
    %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh_1, [{}, {}]>]>} : tensor<8x8xf32>
    return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @multiple_shardings_with_device_list(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[2,4]0,4,1,5,2,6,3,7}"},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,4,2]0,4,2,6,1,5,3,7 last_tile_dim_replicate}"},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[1,2,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}"})
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @multiple_shardings_with_device_list(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"axis_2"}, {"axis_0", "axis_1"}]>},
                              %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"axis_0", "axis_2"}]>},
                              %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"axis_1"}]>}) -> tensor<8x16xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: {mhlo.sharding = "{devices=[4,1,2]0,2,1,3,4,6,5,7 last_tile_dim_replicate}"}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"axis_1","axis_0"}, {}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.dot %0, %arg2 : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @named_computation_in_manual_computation_partially_manual(
// CHECK-SAME:      %arg0: tensor<32x2xi32> {mhlo.sharding = "{devices=[32,1]<=[32]}"})
// CHECK-SAME:      -> (tensor<32x2xi32> {mhlo.sharding = "{devices=[32,1]<=[32]}"}) {
func.func @named_computation_in_manual_computation_partially_manual(
      %arg0: tensor<32x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}, {}]>})
      -> (tensor<32x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}, {}]>}) {
  // CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[32,1]<=[32]}"} : tensor<32x2xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"} : (tensor<32x2xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: %[[FOO:.*]] = call @foo(%[[FULL_TO_SHARD]]) {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"} : (tensor<4x2xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[FOO]] {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"} : tensor<4x2xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<4x2xi32>) -> tensor<32x2xi32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<32x2xi32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_2, [{"x", "y"}, {}]>] out_shardings=[<@mesh_2, [{"x", "y"}, {}]>] manual_axes={"x"} (%arg1: tensor<4x2xi32>) {
    %1 = sdy.named_computation<"foo">(%arg1) in_shardings=[<@mesh_2, [{"y"}, {}]>] out_shardings=[<@mesh_2, [{"y"}, {}]>] (%arg2: tensor<4x2xi32>) {
      %2 = stablehlo.multiply %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"y"}, {}]>]>} : tensor<4x2xi32>
      sdy.return %2 : tensor<4x2xi32>
    } : (tensor<4x2xi32>) -> tensor<4x2xi32>
    sdy.return %1 : tensor<4x2xi32>
  } : (tensor<32x2xi32>) -> tensor<32x2xi32>
  return %0 : tensor<32x2xi32>
}

// CHECK-LABEL: func @named_computation_in_manual_computation_fully_manual(
// CHECK-SAME:      %arg0: tensor<32xi32> {mhlo.sharding = "{devices=[32]<=[32]}"})
// CHECK-SAME:      -> (tensor<32xi32> {mhlo.sharding = "{devices=[32]<=[32]}"}) {
func.func @named_computation_in_manual_computation_fully_manual(
      %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}]>})
      -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}]>}) {
  // CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[32]<=[32]}"} : tensor<32xi32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<32xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[FOO:.*]] = call @foo_0(%[[FULL_TO_SHARD]]) {mhlo.sharding = "{manual}"} : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[FOO]] {mhlo.sharding = "{manual}"} : tensor<1xi32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]]) {mhlo.sharding = "{devices=[32]<=[32]}"} : (tensor<1xi32>) -> tensor<32xi32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<32xi32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_2, [{"x", "y"}]>] out_shardings=[<@mesh_2, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<1xi32>) {
    %1 = sdy.named_computation<"foo">(%arg1) out_shardings=[<@mesh_2, [{}]>] (%arg2: tensor<1xi32>) {
      %2 = stablehlo.negate %arg2 : tensor<1xi32>
      sdy.return %2 : tensor<1xi32>
    } : (tensor<1xi32>) -> tensor<1xi32>
    sdy.return %1 : tensor<1xi32>
  } : (tensor<32xi32>) -> tensor<32xi32>
  return %0 : tensor<32xi32>
}

// CHECK-LABEL: func @free_axis_inside_in_out_shardings_manual_computation
func.func @free_axis_inside_in_out_shardings_manual_computation(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i", ?}, {?}]>}) {
  // CHECK-NEXT: %[[COPY_OPERAND:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND]]) {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dims={manual}}"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[MULT:.*]] = stablehlo.multiply %[[FULL_TO_SHARD]], %[[FULL_TO_SHARD]] {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dims={manual}}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %[[MULT]] {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dims={manual}}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[COPY_RESULT:.*]] = mhlo.copy %[[COPY]] {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dims={manual}}"} : tensor<4x8xf32>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT]]) {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_5, [{"i", ?}, {?}], replicated={"j"}>]
      out_shardings=[<@mesh_5, [{"i", ?}, {?}], replicated={"j"}>]
      manual_axes={"j"} (%arg1: tensor<4x8xf32>) {
    %1 = stablehlo.multiply %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{"i"}, {}]>]>} : tensor<4x8xf32>
    %2 = sdy.reshard %1 <@mesh_5, [{"i"}, {}]> : tensor<4x8xf32>
    sdy.return %2 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_erf_topk
func.func @custom_call_erf_topk(
  %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i"}, {}]>}
  ) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i", ?}, {?}]>}) {
  // CHECK-NEXT: %[[ERF:.*]] = stablehlo.custom_call @mhlo.erf(%arg0) {mhlo.attributes = {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}", mhlo.version = 1 : i64}} : (tensor<16x8xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: stablehlo.custom_call @mhlo.topk(%[[ERF]])
  // CHECK-SAME{LITERAL}: {mhlo.attributes = {k = 2 : i64, largest = true, mhlo.sharding = "{{devices=[2,1,2]<=[4] last_tile_dim_replicate}, {devices=[2,1,2]<=[4] last_tile_dim_replicate}}"}, mhlo.version = 1 : i64} : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  %0 = stablehlo.custom_call @mhlo.erf(%arg0) {
    mhlo.attributes = {mhlo.version = 1 : i64},
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{"i", ?}, {?}]>]>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %1:2 = stablehlo.custom_call @mhlo.topk(%0) {
    mhlo.attributes = {k = 2 : i64, largest = true},
    mhlo.version = 1 : i64,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{"i", ?}, {?}]>, <@mesh_5, [{"i", ?}, {?}]>]>
  } : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %1#0 : tensor<16x2xf32>
}

// CHECK-LABEL: @callback_transform_to_tuple
func.func @callback_transform_to_tuple(%arg0: tensor<2xf64> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i"}]>}) -> (tensor<2xf64> {sdy.sharding = #sdy.sharding<@mesh_5, [{"i"}]>}) {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[CALLBACK:.*]] = stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0) {{{.*}} : (tensor<i64>, tensor<2xf64>) -> tuple<tensor<2xf64>>
  // CHECK-NEXT: %[[GET_TUPLE:.*]] = stablehlo.get_tuple_element %[[CALLBACK]][0] {mhlo.sharding = "{replicated}"} : (tuple<tensor<2xf64>>) -> tensor<2xf64>
  // CHECK-NEXT: return %[[GET_TUPLE]] : tensor<2xf64>
  %1 = stablehlo.constant dense<56560393354880> : tensor<i64>
  %2 = stablehlo.custom_call @xla_python_cpu_callback(%1, %arg0) {api_version = 2 : i32, backend_config = "56560393354880", operand_layouts = [dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh_0, [{}]>]>, xla_shape = "(f64[2]{0})"} : (tensor<i64>, tensor<2xf64>) -> tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: @callback_no_result
func.func private @callback_no_result(%arg0: tensor<f64>) {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0) {
  // CHECK-SAME:   api_version = 2 : i32, backend_config = "56238273106176",
  // CHECK-SAME:   has_side_effect = true,  mhlo.sharding = "{maximal device=0}",
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>],
  // CHECK-SAME: } : (tensor<i64>, tensor<f64>) -> ()
  %c = stablehlo.constant dense<56238273106176> : tensor<i64>
  stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {api_version = 2 : i32, backend_config = "56238273106176", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [], sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (tensor<i64>, tensor<f64>) -> ()
  return
}

// CHECK-LABEL: @callback_tuple_result_token_used
func.func public @callback_tuple_result_token_used(%arg0: !stablehlo.token, %arg1: tensor<2xi64>) -> !stablehlo.token {
  %c = stablehlo.constant dense<56238119409280> : tensor<i64>
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[CALLBACK:.*]] = stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0, %arg1) {
  // CHECK-SAME:   api_version = 2 : i32, backend_config = "56238119409280",
  // CHECK-SAME:   has_side_effect = true, mhlo.sharding = "{maximal device=0}",
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>],
  // CHECK-SAME:   result_layouts = [dense<> : tensor<0xindex>]
  // CHECK-SAME: } : (tensor<i64>, !stablehlo.token, tensor<2xi64>) -> tuple<!stablehlo.token>
  // CHECK-NEXT: %[[TOKEN:.*]] = stablehlo.get_tuple_element %[[CALLBACK]][0] : (tuple<!stablehlo.token>) -> !stablehlo.token
  // CHECK-NEXT: return %[[TOKEN]] : !stablehlo.token
  %0 = stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0, %arg1) {api_version = 2 : i32, backend_config = "56238119409280", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<> : tensor<0xindex>], sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (tensor<i64>, !stablehlo.token, tensor<2xi64>) -> tuple<!stablehlo.token>
  %1 = stablehlo.get_tuple_element %0[0] : (tuple<!stablehlo.token>) -> !stablehlo.token
  return %1 : !stablehlo.token
}

// CHECK-LABEL: @callback_no_tuple_result_used
func.func @callback_no_tuple_result_used(%arg0: tensor<2xf64>) -> tensor<2xf64> {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: %[[CALLBACK:.*]] = stablehlo.custom_call @xla_python_cpu_callback(%[[C]], %arg0) {{{.*}} : (tensor<i64>, tensor<2xf64>) -> tuple<tensor<2xf64>>
  // CHECK-NEXT: %[[GET_TUPLE:.*]] = stablehlo.get_tuple_element %[[CALLBACK]][0] {mhlo.sharding = "{replicated}"} : (tuple<tensor<2xf64>>) -> tensor<2xf64>
  // CHECK-NEXT: return %[[GET_TUPLE]] : tensor<2xf64>
  %c = stablehlo.constant dense<18990036333952> : tensor<i64>
  %0 = stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {api_version = 2 : i32, backend_config = "18990036333952", operand_layouts = [dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>], sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh_0, [{?}]>]>, xla_shape = "(f64[2]{0})"} : (tensor<i64>, tensor<2xf64>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func @maximal_sharding_no_results
// CHECK-SAME:      (%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @maximal_sharding_no_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @foo(%arg0) {has_side_effect = true, mhlo.sharding = "{maximal device=0}"} : (tensor<8x8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  stablehlo.custom_call @foo(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @while_with_sharding
func.func @while_with_sharding(
    %arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_1 = %[[C0]])
  // CHECK-SAME{LITERAL}: attributes {mhlo.sharding = "{{devices=[8,1,4]<=[32] last_tile_dim_replicate}, {replicated}}"}
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32> attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>, <@mesh_2, []>]>}
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    stablehlo.return %iterArg, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// CHECK-LABEL: func @while_with_no_sharding
func.func @while_with_no_sharding(
    %arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<0>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_1 = %[[C0]])
  // CHECK-SAME: attributes {mhlo.sharding = "{replicated}"}
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare LT, %iterArg_1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    stablehlo.return %iterArg, %iterArg_1 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// CHECK-LABEL: func @while_with_no_sharding_inside_manual_comp
func.func @while_with_no_sharding_inside_manual_comp(
      %arg0: tensor<32x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}, {}]>})
      -> (tensor<32x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x", "y"}, {}]>}) {
  // CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[32,1]<=[32]}"}
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_0]])
  // CHECK:      %[[C0:.*]] = stablehlo.constant {mhlo.sharding = "{manual}"} dense<0>
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[FULL_TO_SHARD]], %iterArg_1 = %[[C0]])
  // CHECK-SAME:   attributes {mhlo.sharding = "{{[{][{]}}manual}, {manual}}"}
  // CHECK:      %[[COPY_1:.*]] = mhlo.copy %[[WHILE]]#0
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_1]])
  // CHECK-NEXT: return %[[SHARD_TO_FULL]]
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_2, [{"x", "y"}, {}]>] out_shardings=[<@mesh_2, [{"x", "y"}, {}]>] manual_axes={"x", "y"} (%arg1: tensor<1x2xi32>) {
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.constant dense<32> : tensor<i32>
    %3:2 = stablehlo.while(%iterArg = %arg1, %iterArg_1 = %1) : tensor<1x2xi32>, tensor<i32>
      cond {
      %4 = stablehlo.compare LT, %iterArg_1, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %4 : tensor<i1>
    } do {
      stablehlo.return %iterArg, %iterArg_1 : tensor<1x2xi32>, tensor<i32>
    }
    sdy.return %3#0 : tensor<1x2xi32>
  } : (tensor<32x2xi32>) -> tensor<32x2xi32>
  return %0 : tensor<32x2xi32>
}

// CHECK-LABEL: func @propagation_barrier
func.func @propagation_barrier(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>) {
  // CHECK-NEXT: return %arg0 : tensor<8x16xf32>
  %r = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8x16xf32>
  return %r : tensor<8x16xf32>
}

// CHECK-LABEL: func @all_reduce_input_no_unreduced_axes
func.func @all_reduce_input_no_unreduced_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"j"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  %0 = sdy.all_reduce {"i"} %arg0 out_sharding=<@mesh_5, [{"j"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_input_with_unreduced_axes
// CHECK-SAME:  (%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[2,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}) -> tensor<8x8xf32> {
// CHECK-NOT:   mhlo.frontend_attributes
func.func @all_reduce_input_with_unreduced_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{"j"}, {}], unreduced={"i"}>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY_0:.*]] = mhlo.copy %arg0 {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}, mhlo.sharding = "{devices=[2,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_0]]) {mhlo.sharding = "{manual}"}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%1) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // CHECK-SAME:   replica_groups = dense<{{\[}}[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg2 {mhlo.sharding = "{manual}"}
  // CHECK-NEXT:   stablehlo.return %[[ADD]]
  // CHECK-NEXT: }) {mhlo.sharding = "{manual}"}
  // CHECK-NEXT: %[[COPY_1:.*]] = mhlo.copy %[[ALL_REDUCE]] {mhlo.sharding = "{manual}"}
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[2,1,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}
  // CHECK-NEXT: return %[[SHARD_TO_FULL]]
  %0 = sdy.all_reduce {"i"} %arg0 out_sharding=<@mesh_5, [{"j"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// Unreduced frontend attribute tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unreduced_func_input
func.func @unreduced_func_input(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{}, {}], unreduced={"j"}>}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg0, %arg1
  // CHECK-NOT:  mhlo.frontend_attributes
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unreduced_op
func.func @unreduced_op(%arg0: tensor<4x64x16xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{}, {"i", "j"}, {}]>}) -> tensor<4x16xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}], unreduced={"i"}>]>} : (tensor<4x64x16xf32>, tensor<f32>) -> tensor<4x16xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}], unreduced={"i"}>]>} : tensor<4x16xf32>
  %3 = sdy.all_reduce {"i"} %2 out_sharding=<@mesh_5, [{}, {}]> : tensor<4x16xf32>
  return %3 : tensor<4x16xf32>
}

// CHECK-LABEL: func @no_unreduced_op
func.func @no_unreduced_op(%arg0: tensor<4x64x16xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{}, {"i", "j"}, {}]>}) -> tensor<4x16xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1]
  // CHECK-NOT:  mhlo.frontend_attributes
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]]
  // CHECK-NOT:  mhlo.frontend_attributes
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}]>]>} : (tensor<4x64x16xf32>, tensor<f32>) -> tensor<4x16xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}]>]>} : tensor<4x16xf32>
  return %2 : tensor<4x16xf32>
}

// CHECK-LABEL: func @both_results_unreduced
func.func @both_results_unreduced(%arg0: tensor<4x64x16xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{}, {"i", "j"}, {}]>}, %arg1: tensor<4x64x16xf32> {sdy.sharding = #sdy.sharding<@mesh_5, [{}, {"i", "j"}, {}]>}) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  // CHECK:      %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %cst), (%arg1 init: %cst) across dimensions = [1] {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}
  // CHECK:      %[[ADD0:.*]] = stablehlo.add %[[REDUCE]]#0, %[[REDUCE]]#0 {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[REDUCE]]#1, %[[REDUCE]]#1 {mhlo.frontend_attributes = {xla.sdy.has_unreduced_axes = "true"}
  // CHECK:      return %[[ADD0]], %[[ADD1]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}], unreduced={"i"}>, <@mesh_5, [{}, {}], unreduced={"i"}>]>} : (tensor<4x64x16xf32>, tensor<4x64x16xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x16xf32>, tensor<4x16xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  %2 = stablehlo.add %1#0, %1#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}], unreduced={"i"}>]>} : tensor<4x16xf32>
  %3 = stablehlo.add %1#1, %1#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_5, [{}, {}], unreduced={"i"}>]>} : tensor<4x16xf32>
  return %2, %3 : tensor<4x16xf32>, tensor<4x16xf32>
}


// CHECK-LABEL: func private @foo
// CHECK-SAME:    %arg0: tensor<4x2xi32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"}
// CHECK-SAME:    -> (tensor<4x2xi32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"}) {
// CHECK-NEXT:    %[[MULT:.*]] = stablehlo.multiply %arg0, %arg0 {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dims={manual}}"} : tensor<4x2xi32>
// CHECK-NEXT:    return %[[MULT]] : tensor<4x2xi32>

// CHECK-LABEL: func private @foo_0
// CHECK-SAME:    %arg0: tensor<1xi32> {mhlo.sharding = "{manual}"}
// CHECK-SAME:    -> (tensor<1xi32> {mhlo.sharding = "{manual}"}) {
// CHECK-NEXT:    %[[NEGATE:.*]] = stablehlo.negate %arg0 {mhlo.sharding = "{manual}"} : tensor<1xi32>
// CHECK-NEXT:    return %[[NEGATE]] : tensor<1xi32>
