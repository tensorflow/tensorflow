// RUN: sdy_opt %s -xla-sdy-stablehlo-export-pipeline='keep-hlo-sharding-constraints=true' -xla-sdy-stablehlo-import-pipeline 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=4, "b"=2]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME    in_shardings=[<@mesh, [{"_axis_0"}, {"_axis_1"}]>, <@mesh, [{"_axis_1"}, {}]>]
  // CHECK-SAME    out_shardings=[<@mesh, [{"_axis_0"}, {}]>]
  // CHECK-SAME    manual_axes={"_axis_0", "_axis_1"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:   %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT:   %2 = stablehlo.dot %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT:   sdy.return %2 : tensor<2x32xf32>
  // CHECK-NEXT: } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %0 : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, [{"b"}, {}], replicated={"a"}>]
      out_shardings=[<@mesh, [{"a"}, {}], replicated={"b"}>]
      manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    sdy.return %2 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"_axis_0"}, {}]>] out_shardings=[<@mesh, [{"_axis_0"}, {}]>] manual_axes={"_axis_0"}
  // CHECK:      %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}, {"_axis_1"}]>] out_shardings=[<@mesh, [{}, {"_axis_1"}]>] manual_axes={"_axis_1"}
  // CHECK.      return %1
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}, {"b"}]>] out_shardings=[<@mesh, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
