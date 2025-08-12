// RUN: sdy_opt %s -xla-sdy-import-shardings -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4, "_axis_2"=4]>

// CHECK-LABEL: func @non_trivial_common_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_2"}, {"_axis_0", "_axis_1"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {"_axis_2"}]>})
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @non_trivial_common_mesh(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
                                   %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,2,16]<=[32] last_tile_dim_replicate}"},
                                   %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,4,2]<=[2,16]T(1,0) last_tile_dim_replicate}"}) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = "stablehlo.dot" (%0, %arg2) : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @multiple_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_2"}, {"_axis_0", "_axis_1"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0", "_axis_2"}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_1"}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0", "_axis_1"}, {"_axis_2"}]>}) {
func.func @multiple_shardings(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"},
                              %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[1,8,4]<=[2,4,4]T(0,2,1) last_tile_dim_replicate}"},
                              %arg2: tensor<8x16xf32> {mhlo.sharding = "{devices=[1,4,8]<=[2,4,4]T(1,0,2) last_tile_dim_replicate}"})
    -> (tensor<8x16xf32> {mhlo.sharding = "{devices=[8,4]<=[32]}"}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"_axis_1", "_axis_0"}, {}]>]>}
  %0 = stablehlo.add %arg0, %arg1 {mhlo.sharding = "{devices=[8,1,4]<=[2,4,4]T(1,0,2) last_tile_dim_replicate}"} : tensor<8x8xf32>
  %1 = "stablehlo.dot" (%0, %arg2) : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=16]>

// CHECK-LABEL: func @single_axis(
// CHECK-SAME:      %arg0: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32>)
// CHECK-SAME:  -> tensor<32x16xf32> {
func.func @single_axis(%arg0: tensor<32x8xf32> {mhlo.sharding = "{devices=[16,1]<=[16]}"},
                       %arg1: tensor<8x16xf32>) -> tensor<32x16xf32> {
  %0 = "stablehlo.dot" (%arg0, %arg1) : (tensor<32x8xf32>, tensor<8x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>

// CHECK-LABEL: func @multi_result_op
func.func @multi_result_op(%arg0: tensor<4x64x8xf32>, %arg1: tensor<4x64x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK: stablehlo.reduce
// CHECK-SAME{LITERAL}: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"_axis_1"}]>, <@mesh, [{"_axis_1"}, {}]>]>}
  %1:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %0) across dimensions = [1]
    {mhlo.sharding = "{{devices=[1,4,8]<=[8,4]T(1,0) last_tile_dim_replicate}, {devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}}"} :
    (tensor<4x64x8xf32>, tensor<4x64x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<f32>, %arg5: tensor<f32>)  {
      %2 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %3 = stablehlo.add %arg3, %arg5 : tensor<f32>
      stablehlo.return %2, %3 : tensor<f32>, tensor<f32>
    }
  return %1#0, %1#1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>

// CHECK-LABEL: func @fully_replicated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32>)
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @fully_replicated(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
                            %arg1: tensor<8x8xf32> {mhlo.sharding = "{replicated}"},
                            %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = "stablehlo.dot" (%0, %arg2) : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=7, "_axis_1"=2, "_axis_2"=5, "_axis_3"=3]>

// CHECK-LABEL: func @prime_number(
// CHECK-SAME:       %arg0: tensor<6x35xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_3", "_axis_1"}, {"_axis_2", "_axis_0"}]>}
// CHECK-SAME:       %arg1: tensor<6x35xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME:    -> tensor<6x35xf32> {
func.func @prime_number(%arg0: tensor<6x35xf32> {mhlo.sharding = "{devices=[6,35]<=[7,10,3]T(2,1,0)}"},
                        %arg1: tensor<6x35xf32> {mhlo.sharding = "{replicated}"}) -> tensor<6x35xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<6x35xf32>
  return %0 : tensor<6x35xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=3, "_axis_2"=5, "_axis_3"=7, "_axis_4"=11]>

// CHECK-LABEL: func @prime_number_2(
// CHECK-SAME:       %arg0: tensor<231x550x42x42xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1", "_axis_4"}, {"_axis_2", "_axis_0"}, {}, {"_axis_3"}]>}
// CHECK-SAME:       %arg1: tensor<231x550x42x42xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_3"}, {"_axis_2", "_axis_4"}, {"_axis_1", "_axis_0"}, {}]>})
// CHECK-SAME:    -> tensor<231x550x42x42xf32> {
func.func @prime_number_2(%arg0: tensor<231x550x42x42xf32> {mhlo.sharding = "{devices=[33,10,1,7]<=[2,3,5,7,11]T(1,4,2,0,3)}"},
                          %arg1: tensor<231x550x42x42xf32> {mhlo.sharding = "{devices=[7,55,6,1]<=[2,3,5,7,11]T(3,2,4,1,0)}"}) -> tensor<231x550x42x42xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<231x550x42x42xf32>
  return %0 : tensor<231x550x42x42xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>

// CHECK-LABEL: func @unknown_sharding(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}]>})
// CHECK-SAME:  -> tensor<8x8xf32> {
func.func @unknown_sharding(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
                            %arg1: tensor<8x8xf32> {mhlo.sharding = "{unknown}"}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <[]>
// CHECK-LABEL: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @one_maximal_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>}
func.func @one_maximal_mesh(%arg0: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"},
                            %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
// CHECK-LABEL: sdy.mesh @maximal_mesh_4 = <[], device_ids=[4]>

// CHECK-LABEL: func @two_maximal_shardings_should_be_sorted(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_4, []>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>})
func.func @two_maximal_shardings_should_be_sorted(%arg0: tensor<8x8xf32> {mhlo.sharding = "{maximal device=4}"},
                            %arg1: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----
// CHECK-COUNT-1: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @duplicate_maximal_sharding_should_be_deduped(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>})
func.func @duplicate_maximal_sharding_should_be_deduped(%arg0: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"},
                            %arg1: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"}) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>
// CHECK-LABEL: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @two_meshes(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>}, %arg2: tensor<8x16xf32>)
func.func @two_meshes(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
                            %arg1: tensor<8x8xf32> {mhlo.sharding = "{maximal device=0}"},
                            %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = "stablehlo.dot" (%0, %arg2) : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----
// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>
// CHECK-LABEL: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @maximal_sharding_on_op(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32>)
// CHECK-SAME:  -> tensor<8x8xf32> {
func.func @maximal_sharding_on_op(%arg0: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"},
                            %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
// CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// CHECK-SAME{LITERAL}: {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_4, []>]>}
// CHECK-NEXT: %[[MULTIPLY:.*]] = stablehlo.multiply %[[ADD]], %[[ADD]]
// CHECK-SAME{LITERAL}: {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>}
  %0 = stablehlo.add %arg0, %arg1 {mhlo.sharding = "{maximal device=4}"} : tensor<8x8xf32>
  %1 = stablehlo.multiply %0, %0 {mhlo.sharding = "{maximal device=0}"} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: sdy.mesh @mesh = <["_axis_0"=8, "_axis_1"=4]>
// CHECK-LABEL: func @import_sharding_with_token_types
// CHECK-SAME{LITERAL}:      %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}
// CHECK-SAME{LITERAL}:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_1"}, {"_axis_0"}]>}
func.func @import_sharding_with_token_types(%arg0: !stablehlo.token {mhlo.sharding = "{replicated}"},
                                           %arg1: tensor<8x8xf32> {mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"})
                                           -> (tensor<f32>, !stablehlo.token) {
// CHECK-NEXT: %[[CALL1:.*]]:2 = stablehlo.custom_call @foo(%arg0, %arg1)
// CHECK-NEXT: %{{.*}} = stablehlo.custom_call @Sharding(%[[CALL1]]#1)
// CHECK-SAME{LITERAL}: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>]>}
  %0:2 = stablehlo.custom_call @foo(%arg0, %arg1) : (!stablehlo.token, tensor<8x8xf32>) -> (tensor<f32>, !stablehlo.token)
  %1 = stablehlo.custom_call @Sharding(%0#1) {mhlo.sharding = "{replicated}"} : (!stablehlo.token) -> !stablehlo.token
  func.return %0#0, %1 : tensor<f32>, !stablehlo.token
}
