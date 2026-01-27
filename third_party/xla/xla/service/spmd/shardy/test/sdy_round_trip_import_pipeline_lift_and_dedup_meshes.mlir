// RUN: sdy_opt %s -xla-sdy-round-trip-import-pipeline='lift-and-dedup-meshes=true' 2>&1 | FileCheck %s

// CHECK-LABEL: module @multiple_inlined_mesh_shardings
module @multiple_inlined_mesh_shardings {
  // CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
  // CHECK: sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
  // CHECK: sdy.mesh @maximal_mesh_2 = <[], device_ids=[2]>

  // CHECK-LABEL: func @inlined_mesh1(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@maximal_mesh_0, []>}) {
  func.func @inlined_mesh1(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"a\"=2, \"b\"=2]>, [{\"a\"}]>"}}
  ) -> tensor<32xi32> {
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a", "b"}]> : tensor<32xi32>
    // CHECK-NEXT: return %[[SHARDING]]
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"c\"=4]>, [{\"c\"}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[], device_ids=[0]>, []>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %1 : tensor<32xi32>
  }

  // CHECK-LABEL: func @inlined_mesh2(
  // CHECK-SAME: %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}]>})
  // CHECK-SAME: -> (tensor<32xi32> {sdy.sharding = #sdy.sharding<@maximal_mesh_2, []>}) {
  func.func @inlined_mesh2(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"a\"=4, \"b\"=1]>, [{\"a\"}]>"}}
  ) -> tensor<32xi32> {
    // CHECK-NEXT: %[[SHARDING:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a", "b"}]> : tensor<32xi32>
    // CHECK-NEXT: return %[[SHARDING]]
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"c\"=4]>, [{\"c\"}]>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    %1 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%0) {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[], device_ids=[2]>, []>]>"}} : (tensor<32xi32>) -> tensor<32xi32>
    return %1 : tensor<32xi32>
  }
}
