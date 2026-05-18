// RUN: sdy_opt %s -xla-sdy-stablehlo-export-manual-reduction-collectives='export-all-reduce-scatter=true' 2>&1 | FileCheck %s
// RUN: sdy_opt %s -xla-sdy-stablehlo-export-manual-reduction-collectives='export-all-reduce-scatter=false' 2>&1 | FileCheck %s --check-prefix=NO-EXPORT

sdy.mesh @mesh = <["x"=8, "y"=4]>
sdy.mesh @mesh_x_4 = <["x"=4]>

// CHECK-LABEL: func @reduce_scatter
// NO-EXPORT-LABEL: func @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"x"}, {}], unreduced={"y"}>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"x"}, {"y"}]>]
  // CHECK-SAME:     manual_axes={"x", "y"} (%arg1: tensor<1x8xf32>) {
  // CHECK-NEXT:   %[[REDUCE_SCATTER:.*]] = "stablehlo.reduce_scatter"(%arg1)
  // CHECK:        sdy.return %[[REDUCE_SCATTER]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[MANUAL_COMP]]

  // NO-EXPORT-NEXT: %[[RS:.*]] = sdy.reduce_scatter [{}, {"y"}] %arg0
  // NO-EXPORT-NEXT: return %[[RS]]

  %0 = sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reduce_scatter_sort_and_merge
func.func @reduce_scatter_sort_and_merge(%arg0: tensor<8x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_x_4, [{}, {}, {}], unreduced={"x"}>}) -> tensor<8x8x8xf32> {
  // CHECK-NEXT: %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:     in_shardings=[<@mesh_x_4, [{}, {}, {}], unreduced={"x"}>]
  // CHECK-SAME:     out_shardings=[<@mesh_x_4, [{"x":(1)2}, {}, {"x":(2)2}]>]
  // CHECK-SAME:     manual_axes={"x"} (%arg1: tensor<8x8x8xf32>) {
  // CHECK:        sdy.return
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[MANUAL_COMP]]

  %0 = sdy.reduce_scatter [{"x":(1)2}, {}, {"x":(2)2}] %arg0 out_sharding=<@mesh_x_4, [{"x":(1)2}, {}, {"x":(2)2}]> : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}
