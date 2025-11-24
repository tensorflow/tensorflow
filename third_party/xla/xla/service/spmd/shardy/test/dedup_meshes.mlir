// RUN: sdy_opt %s -xla-sdy-round-trip-dedup-meshes 2>&1 | FileCheck %s

// CHECK:     sdy.mesh @mesh1 = <["a"=2, "b"=4]>
// CHECK-NOT: sdy.mesh @mesh2 = <["data"=2, "model"=4]>
// CHECK-NOT: sdy.mesh @mesh3 = <["x"=2, "y"=4]>
// CHECK:     sdy.mesh @mesh4 = <["x"=2, "y"=4], device_ids=[7, 1, 2, 3, 4, 5, 6, 0]>
// CHECK-NOT: sdy.mesh @mesh5 = <["a"=2, "b"=2, "c"=2], device_ids=[7, 1, 2, 3, 4, 5, 6, 0]>
sdy.mesh @mesh1 = <["a"=2, "b"=4]>
sdy.mesh @mesh2 = <["data"=2, "model"=4]>
sdy.mesh @mesh3 = <["x"=2, "y"=4]>
sdy.mesh @mesh4 = <["x"=2, "y"=4], device_ids=[7, 1, 2, 3, 4, 5, 6, 0]>
sdy.mesh @mesh5 = <["a"=2, "b"=2, "c"=2], device_ids=[7, 1, 2, 3, 4, 5, 6, 0]>

// CHECK:     sdy.mesh @meshB = <["data"=4, "model"=4]>
// CHECK-NOT: sdy.mesh @meshA = <["a"=4, "b"=4]>
sdy.mesh @meshB = <["data"=4, "model"=4]>
sdy.mesh @meshA = <["a"=4, "b"=4]>

// CHECK:     sdy.mesh @mesh_one_device_before_empty = <["x"=1, "y"=1]>
// CHECK:     sdy.mesh @empty_mesh1 = <[]>
// CHECK-NOT: sdy.mesh @empty_mesh2 = <[]>
sdy.mesh @mesh_one_device_before_empty = <["x"=1, "y"=1]>
sdy.mesh @empty_mesh1 = <[]>
sdy.mesh @empty_mesh2 = <[]>
// CHECK:     sdy.mesh @maximal_mesh1 = <[], device_ids=[0]>
// CHECK-NOT: sdy.mesh @maximal_mesh2 = <[], device_ids=[0]>
sdy.mesh @maximal_mesh1 = <[], device_ids=[0]>
sdy.mesh @maximal_mesh2 = <[], device_ids=[0]>

// CHECK:     sdy.mesh @meshC = <["x"=4]>
// CHECK-NOT: sdy.mesh @meshD = <["a"=2, "b"=2]>
sdy.mesh @meshC = <["x"=4]>
sdy.mesh @meshD = <["a"=2, "b"=2]>

// CHECK:     sdy.mesh @mesh_with_size_1 = <["a"=8, "b"=1, "c"=8]>
// CHECK-NOT: sdy.mesh @mesh_with_size_1_dup = <["a"=2, "b"=4, "c"=8]>
sdy.mesh @mesh_with_size_1 = <["a"=8, "b"=1, "c"=8]>
sdy.mesh @mesh_with_size_1_dup = <["a"=2, "b"=4, "c"=8]>

// CHECK:     sdy.mesh @meshE = <["a"=8, "b"=4]>
// CHECK-NOT: sdy.mesh @meshF = <["x"=2, "y"=4, "z"=4]>
// CHECK-NOT: sdy.mesh @meshG = <["x"=32]>
// CHECK-NOT: sdy.mesh @meshH = <["x"=4, "y"=4, "z"=2]>
// CHECK-NOT: sdy.mesh @meshI = <["a"=4, "b"=1, "c"=2, "d"=1, "e"=4, "f"=1]>
sdy.mesh @meshE = <["a"=8, "b"=4]>
sdy.mesh @meshF = <["x"=2, "y"=4, "z"=4]>
sdy.mesh @meshG = <["x"=32]>
sdy.mesh @meshH = <["x"=4, "y"=4, "z"=2]>
sdy.mesh @meshI = <["a"=4, "b"=1, "c"=2, "d"=1, "e"=4, "f"=1]>
sdy.mesh @meshJ = <["x"=2, "y"=16]>

// CHECK-NOT: sdy.mesh @mesh_fake = <["_a"=3, "_b"=2, "_c"=2]>
// CHECK:     sdy.mesh @mesh_main_replace_fake = <["x"=3, "y"=4]>
sdy.mesh @mesh_fake = <["_a"=3, "_b"=2, "_c"=2]>
sdy.mesh @mesh_main_replace_fake = <["x"=3, "y"=4]>

// CHECK:     sdy.mesh @mesh_one_in_middle = <["a"=2, "b"=1, "c"=5]>
// CHECK-NOT: sdy.mesh @mesh_one_in_middle_dup = <["x"=2, "y"=1, "z"=5]>
// CHECK-NOT: sdy.mesh @mesh_one_in_middle_dup_trailing_ones = <["x"=10, "y"=1, "z"=1]>
// CHECK-NOT: sdy.mesh @mesh_one_in_middle_dup_leading_ones = <["a"=1, "b"=1, "c"=2, "d"=5]>
// CHECK-NOT: sdy.mesh @mesh_one_in_middle_dup_no_size_one = <["x"=2, "y"=5]>
sdy.mesh @mesh_one_in_middle = <["a"=2, "b"=1, "c"=5]>
sdy.mesh @mesh_one_in_middle_dup = <["x"=2, "y"=1, "z"=5]>
sdy.mesh @mesh_one_in_middle_dup_trailing_ones = <["x"=10, "y"=1, "z"=1]>
sdy.mesh @mesh_one_in_middle_dup_leading_ones = <["a"=1, "b"=1, "c"=2, "d"=5]>
sdy.mesh @mesh_one_in_middle_dup_no_size_one = <["x"=2, "y"=5]>

// CHECK:     sdy.mesh @main_mesh_trailing_ones = <["a"=3, "b"=2, "c"=1, "d"=1]>
// CHECK-NOT: sdy.mesh @mesh_trailing_ones_dup = <["a"=3, "b"=1, "c"=2]>
sdy.mesh @main_mesh_trailing_ones = <["a"=3, "b"=2, "c"=1, "d"=1]>
sdy.mesh @mesh_trailing_ones_dup = <["a"=3, "b"=1, "c"=2]>

// CHECK:     sdy.mesh @main_mesh_leading_ones = <["a"=1, "b"=1, "c"=3, "d"=3]>
// CHECK-NOT: sdy.mesh @mesh_leading_ones_dup = <["a"=9, "b"=1, "c"=1, "d"=1]>
sdy.mesh @main_mesh_leading_ones = <["a"=1, "b"=1, "c"=3, "d"=3]>
sdy.mesh @mesh_leading_ones_dup = <["a"=9, "b"=1, "c"=1, "d"=1]>

sdy.mesh @mesh_two_small_axes = <["a"=2, "b"=2, "c"=7]>
sdy.mesh @mesh_one_large_axis = <["x"=4, "b"=7]>

sdy.mesh @mesh_large_first_axis = <["a"=16, "b"=2, "c"=7]>
sdy.mesh @mesh_mapping_consecutive_sub_axes = <["x"=4, "y"=8, "z"=7]>

// CHECK-LABEL: @full_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", ?}, {?}]>}
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b", ?}, {?}], unreduced={"a"}>}) {
func.func @full_axes(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", ?}, {?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"model", ?}, {?}], unreduced={"data"}>}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@mesh1, [{"a", ?}p1, {}], replicated={"b"}>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3, [{"x", ?}p1, {}], replicated={"y"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axes
func.func @sub_axes(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@meshB, [{"model":(1)2, ?}p1, {}]>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"b":(1)2, ?}p1, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: @inlined_mesh
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}
func.func @inlined_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @different_device_ids
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh4, [{"x"}, {}]>}
func.func @different_device_ids(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh4, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @empty_mesh
// CHECK-SAME:  (%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh1, [{}, {}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh1, [{}, {}]>}) {
func.func @empty_mesh(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh1, [{}, {}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh2, [{}, {}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @maximal_mesh
// CHECK-SAME:  (%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh1, []>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh1, []>}) {
func.func @maximal_mesh(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh1, []>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@maximal_mesh2, []>}) {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @all_axes_to_sub_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshC, [{"x":(2)2}, {"x":(1)2}]>}
func.func @all_axes_to_sub_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshD, [{"b"}, {"a"}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @partial_axes_to_sub_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh4, [{"x"}, {"y":(1)2}]>}
func.func @partial_axes_to_sub_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh5, [{"a"}, {"b"}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axis_to_sub_axis
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a":(4)2}, {}]>}
func.func @sub_axis_to_sub_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshF, [{"y":(2)2}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @axis_to_multiple_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a", "b"}, {}]>}
func.func @axis_to_multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshG, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axes_to_multiple_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a":(1)4}, {"b":(2)2}]>}
func.func @sub_axes_to_multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshG, [{"x":(1)4}, {"x":(16)2}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axis_to_full_axis
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_two_small_axes, [{"b"}, {}]>}
func.func @sub_axis_to_full_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_large_axis, [{"x":(2)2}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @one_sub_axis_to_multiple_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_large_first_axis, [{"a":(4)4, "b"}, {}]>}
func.func @one_sub_axis_to_multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_mapping_consecutive_sub_axes, [{"y"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axis_to_partial_target_axis
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_large_first_axis, [{"a":(8)2, "b"}, {}]>}
func.func @sub_axis_to_partial_target_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_mapping_consecutive_sub_axes, [{"y":(2)4}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axis_to_second_axis
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a":(4)2}, {"b"}]>}
func.func @sub_axis_to_second_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshH, [{"y":(1)2}, {"y":(2)2, "z"}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @sub_axis_to_sub_and_full_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{}, {"a":(4)2, "b":(1)2}]>}
func.func @sub_axis_to_sub_and_full_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshJ, [{}, {"y":(2)4}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @target_axis_size_one
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a"}, {"b"}, {}]>}
func.func @target_axis_size_one(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@meshI, [{"a", "b", "c"}, {"d", "e"}, {"f"}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_one_in_middle_same_as_main
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle, [{"a"}, {"c"}, {}]>}
func.func @target_mesh_one_in_middle_same_as_main(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle_dup, [{"x", "y"}, {"z"}, {}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_one_in_middle_trailing_ones
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle, [{"a", "c"}, {}, {}]>}
func.func @target_mesh_one_in_middle_trailing_ones(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle_dup_trailing_ones, [{"x", "y"}, {}, {"z"}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_one_in_middle_leading_ones
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle, [{"c"}, {"a"}, {}]>}
func.func @target_mesh_one_in_middle_leading_ones(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle_dup_leading_ones, [{"a", "d"}, {"b", "c"}, {}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_one_in_middle_no_size_one
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle, [{"c"}, {}, {}]>}
func.func @target_mesh_one_in_middle_no_size_one(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_one_in_middle_dup_no_size_one, [{"y"}, {}, {}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_trailing_ones
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@main_mesh_trailing_ones, [{"a"}, {}, {"b"}]>}
func.func @target_mesh_trailing_ones(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_trailing_ones_dup, [{"a"}, {"b"}, {"c"}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @target_mesh_leading_ones
// CHECK-SAME:  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@main_mesh_leading_ones, [{"c", "d"}, {}, {}]>}
func.func @target_mesh_leading_ones(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_leading_ones_dup, [{"a", "d"}, {"b"}, {"c"}]>}) -> tensor<8x8x8xf32> {
  return %arg0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: @fake_axis_to_real_axis
// CHECK-SAME:  %arg0: tensor<3x8xf32> {sdy.sharding = #sdy.sharding<@mesh_main_replace_fake, [{"x"}, {"y":(1)2}]>}
func.func @fake_axis_to_real_axis(%arg0: tensor<3x8xf32> {sdy.sharding = #sdy.sharding<@mesh_fake, [{"_a"}, {"_b"}]>}) -> tensor<3x8xf32> {
  return %arg0 : tensor<3x8xf32>
}

// CHECK-LABEL: @merge_sub_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshC, [{"x"}, {}]>}
func.func @merge_sub_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshD, [{"a", "b"}, {}]>}) -> tensor<8x8xf32> {
  return %arg0 : tensor<8x8xf32>
}

// CHECK-LABEL: @full_axes_merge_sub_axes
// CHECK-SAME:  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a", ?}, {?}]>}
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"b", ?}, {?}], unreduced={"a"}>}) {
func.func @full_axes_merge_sub_axes(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshE, [{"a", ?}, {?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@meshF, [{"z", ?}, {?}], unreduced={"x", "y"}>}) {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@meshE, [{"b", ?}p1, {}], replicated={"a"}>]>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@meshF, [{"z", ?}p1, {}], replicated={"x", "y"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NEXT:          %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@meshE, [{"a":(2)4}, {"a":(1)2}]>, <@meshE, [{"a":(1)2}, {}], replicated={"a":(2)4}>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@meshE, [{"a":(2)4}, {}], replicated={"a":(1)2}>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a"}
  // CHECK-SAME:              (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@meshE, [{?}, {"b", ?}]>]>}
  // CHECK:               } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:          return %[[MAN_COMP]]
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@meshF, [{"y"}, {"x"}]>, <@meshF, [{"x"}, {}], replicated={"y"}>]
      out_shardings=[<@meshF, [{"y"}, {}], replicated={"x"}>]
      manual_axes={"x", "y"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@meshE, [{?}, {"b", ?}]>]>} : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    sdy.return %2 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_computation_inlined_meshes
func.func @manual_computation_inlined_meshes(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NEXT:          sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<mesh<["x"=2, "y"=4, "z"=4]>, [{"y"}, {"x"}]>, <mesh<["x"=2, "y"=4, "z"=4]>, [{"x"}, {}], replicated={"y"}>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<mesh<["x"=2, "y"=4, "z"=4]>, [{"y"}, {}], replicated={"x"}>]
  // CHECK-SAME{LITERAL}:     manual_axes={"x", "y"}
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<mesh<["x"=2, "y"=4, "z"=4]>, [{"y"}, {"x"}]>, <mesh<["x"=2, "y"=4, "z"=4]>, [{"x"}, {}], replicated={"y"}>]
      out_shardings=[<mesh<["x"=2, "y"=4, "z"=4]>, [{"y"}, {}], replicated={"x"}>]
      manual_axes={"x", "y"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@meshE, [{?}, {"b", ?}]>]>} : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    sdy.return %2 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
