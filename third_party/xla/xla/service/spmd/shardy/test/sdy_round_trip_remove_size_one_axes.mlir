// RUN: sdy_opt %s -xla-sdy-round-trip-remove-size-one-axes 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1], device_ids=[0, 2, 1, 3, 4, 6, 5, 7]>
sdy.mesh @mesh2 = <["a"=4, "b"=2]>
sdy.mesh @mesh3 = <["x"=1, "y"=1]>
sdy.mesh @mesh4 = <["a"=1, "b"=2, "c"=1]>

// CHECK: sdy.mesh @mesh1 = <["a"=1, "b"=2, "c"=1, "d"=4, "e"=1], device_ids=[0, 2, 1, 3, 4, 6, 5, 7]>
// CHECK: sdy.mesh @mesh2 = <["a"=4, "b"=2]>
// CHECK: sdy.mesh @mesh3 = <["x"=1, "y"=1]>
// CHECK: sdy.mesh @mesh4 = <["a"=1, "b"=2, "c"=1]>

// CHECK-LABEL: func @func_and_op_shardings
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b"}, {?}]>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"d", ?}, {}], replicated={"b"}>},
// CHECK-SAME:    %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a"}, {"b"}]>}
// CHECK-SAME:  ) -> (
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{}, {?}], unreduced={"d"}>},
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b"}, {}]>}) {
func.func @func_and_op_shardings(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", "b"}, {"c", ?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"d", "e", ?}, {}], replicated={"b", "c"}>},
  %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a"}, {"b"}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"e"}, {"c", ?}], unreduced={"a", "d"}>},
      tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", "b", "c"}, {}], unreduced={"e"}>}) {
  // CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}, {?}]>]>}
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add %arg2, %arg2
  // CHECK-NOT:    sdy.sharding
  // CHECK-NEXT:   %[[ADD3:.*]] = stablehlo.add %[[ADD2]], %[[ADD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{}, {}], replicated={"d"}>]>}
  // CHECK-NEXT:   return %[[ADD1]], %[[ADD3]]
  // CHECK-NEXT: }
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}, {"e", ?}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %arg2, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"c"}, {}], replicated={"d"}>]>} : tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @inlined_mesh
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=1, "b"=2, "c"=1, "d"=2], device_ids=[3, 1, 2, 0]>, [{"b"}, {?}]>}
// CHECK-SAME:  ) -> (
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a", "b"}, {}]>}) {
func.func @inlined_mesh(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=1, "b"=2, "c"=1, "d"=2], device_ids=[3, 1, 2, 0]>, [{"a", "b"}, {"c", ?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a", "b"}, {}]>}) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=1, "b"=1]>, [{?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["a"=1, "b"=1]>, [{"a", ?}, {"b", ?}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @shardings_with_priorities
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"b"}p0, {?}p3], replicated={"d"}>},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a", ?}p2, {}]>}
// CHECK-SAME:  ) -> (
// CHECK-SAME:    tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3, [{}, {?}p2]>}) {
func.func @shardings_with_priorities(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh1, [{"a", "b"}p0, {"c", ?}p3], replicated={"d", "e"}>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"a", ?}p2, {}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3, [{"x"}p1, {"y", ?}p2]>}) {
  // CHECK-NEXT:   %[[ADD1:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}p1, {?}]>]>}
  // CHECK-NEXT:   %[[ADD2:.*]] = stablehlo.add %[[ADD1]], %[[ADD1]]
  // CHECK-NOT:    sdy.sharding
  // CHECK-NEXT:   return %[[ADD2]]
  // CHECK-NEXT: }
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"d", ?}p1, {"e", ?}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NEXT:          %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh1, [{"d"}, {"b"}]>, <@mesh1, [{"b"}, {}], replicated={"d"}>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh1, [{"d"}, {}], replicated={"b"}>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a", "b", "c", "d"}
  // CHECK-SAME:              (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:            stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{?}, {?}]>]>}
  // CHECK:               } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:          return %[[MAN_COMP]]
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh1, [{"d", "a"}, {"b"}]>, <@mesh1, [{"b"}, {"c", "a"}], replicated={"d"}>]
      out_shardings=[<@mesh1, [{"d"}, {}], replicated={"b", "c"}>]
      manual_axes={"a", "b", "c", "d"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{?}, {"e", ?}]>]>} : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) {
      replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
    } : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_computation_inlined_mesh
func.func @manual_computation_inlined_mesh(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT:          %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh4, [{"b"}, {}]>, <mesh<["a"=1, "b"=2, "c"=1]>, [{"b"}, {}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<mesh<["a"=1, "b"=2, "c"=1]>, [{"b"}, {}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={"a", "b"}
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh4, [{"b", "a"}, {}]>, <mesh<["a"=1, "b"=2, "c"=1]>, [{"b"}, {}], replicated={"a"}>]
      out_shardings=[<mesh<["a"=1, "b"=2, "c"=1]>, [{"a", "b"}, {}]>]
      manual_axes={"a", "b"} (%arg2: tensor<4x16xf32>, %arg3: tensor<4x16xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4x16xf32>
    sdy.return %1 : tensor<4x16xf32>
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
