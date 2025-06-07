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
// CHECK:     sdy.mesh @mesh_one_device_after_empty = <["a"=1, "b"=1, "c"=1]>
sdy.mesh @mesh_one_device_before_empty = <["x"=1, "y"=1]>
sdy.mesh @empty_mesh1 = <[]>
sdy.mesh @mesh_one_device_after_empty = <["a"=1, "b"=1, "c"=1]>
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
// CHECK-NOT: sdy.mesh @meshF = <["x"=4, "y"=2, "z"=2]>
sdy.mesh @meshE = <["a"=8, "b"=4]>
sdy.mesh @meshF = <["x"=2, "y"=4, "z"=4]>

// CHECK-NOT: sdy.mesh @mesh_fake = <["_a"=3, "_b"=2, "_c"=2]>
// CHECK:     sdy.mesh @mesh_main_replace_fake = <["x"=3, "y"=4]>
sdy.mesh @mesh_fake = <["_a"=3, "_b"=2, "_c"=2]>
sdy.mesh @mesh_main_replace_fake = <["x"=3, "y"=4]>

// CHECK:     sdy.mesh @mesh_with_axis_of_size_1_main = <["a"=2, "b"=1, "c"=5]>
// CHECK-NOT: sdy.mesh @mesh_with_axis_of_size_1_dup = <["x"=2, "y"=1, "z"=5]>
sdy.mesh @mesh_with_axis_of_size_1_main = <["a"=2, "b"=1, "c"=5]>
sdy.mesh @mesh_with_axis_of_size_1_dup = <["x"=2, "y"=1, "z"=5]>

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

// CHECK-LABEL: @manual_computation
func.func @manual_computation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@meshB, [{"data"}, {?}]>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@meshB, [{"model":(1)2, ?}, {?}]>]
  // CHECK-SAME{LITERAL}: manual_axes={} (%arg1: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a"}, {?}]>] out_shardings=[<@meshB, [{"model":(1)2, ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    sdy.return %arg1 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
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
