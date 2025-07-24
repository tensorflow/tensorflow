// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-export-pipeline='keep-meshes-inlined=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @inlined_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22x\22}, {}]>"}
// CHECK-SAME:      %arg1: tensor<8x16xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\22x\22=2, \22y\22=2]>, [{}, {\22y\22}]>"}
// CHECK-SAME:  -> tensor<8x16xf32> {
func.func @inlined_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
                        %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{}, {"y"}]>}) -> tensor<8x16xf32> {
// CHECK-NEXT: stablehlo.dot
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\22x\22=2, \22y\22=2]>, [{\22x\22}, {\22y\22}]>]>"}
  %1 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<mesh<["x"=2, "y"=2]>, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
