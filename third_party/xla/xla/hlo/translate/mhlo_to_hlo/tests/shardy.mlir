// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Shouldn't fail with an `attribute created with unregistered dialect` error.

// CHECK-LABEL: HloModule
module {
  sdy.mesh @empty_mesh = <[]>
  // TODO(b/435663161) - Allow Shardy attributes to be lowered via hlo-translate
  // CHECK: f32[1] parameter(0)
  func.func @main(%arg0 : tensor<1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> tensor<1xf32> {
    %0 = mhlo.add %arg0, %arg0 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

// -----

// CHECK-LABEL: HloModule sdy_frontend_attributes{{.*}}frontend_attributes={xla.sdy.meshes={mesh = #sdy.mesh<["x"=2, "y"=4, "z"=4]>}
module @sdy_frontend_attributes attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
      "{mesh = #sdy.mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>}"
    }} {
      func.func @main(
        %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\"x\"}, {}]>"}},
        %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\"y\"}]>"}}
      ) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
      // CHECK: %Arg_0.1 = f32[8,8] parameter(0)
      // CHECK-SAME: sharding={devices=[2,1,16]<=[32] last_tile_dim_replicate}
      // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{\"x\"}, {}]>"}
      // CHECK-NEXT: %Arg_1.1 = f32[8,8] parameter(1)
      // CHECK-SAME: sharding={devices=[1,4,8]<=[2,4,4]T(1,0,2) last_tile_dim_replicate}
      // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding<@mesh, [{}, {\"y\"}]>"}
        %0 = mhlo.add %arg0, %arg1
          {mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"x\"}, {}]>]>"
          }}
          : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %add.1 = f32[8,8] add(%Arg_0.1, %Arg_1.1)
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{\"x\"}, {}]>]>"}
        %1 = "mhlo.custom_call"(%0) {
          call_target_name = "xla.sdy.FuncResultSharding",
          mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"x\", \"y\", ?}, {\"z\"}]>]>"
          }
        } : (tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %[[CUSTOM_CALL_0:.*]] = f32[8,8] custom-call(%add.1)
        // CHECK-SAME: custom_call_target="xla.sdy.FuncResultSharding"
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{\"x\", \"y\", ?}, {\"z\"}]>]>"}
        // CHECK-NEXT: %[[RESHAPE_0:.*]] = f32[8,8] reshape(%[[CUSTOM_CALL_0]])
        // CHECK-SAME: sharding={devices=[8,4]<=[32]}
        %2 = "mhlo.custom_call"(%0) {
          call_target_name = "xla.sdy.FuncResultSharding",
          mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
          }
        } : (tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %[[CUSTOM_CALL_1:.*]] = f32[8,8] custom-call(%add.1)
        // CHECK-SAME: custom_call_target="xla.sdy.FuncResultSharding"
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"}
        // CHECK-NEXT: %[[RESHAPE_1:.*]] = f32[8,8] reshape(%[[CUSTOM_CALL_1]])
        // CHECK-SAME: sharding={replicated}
        return %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>
        // CHECK: ROOT %tuple.1 = (f32[8,8], f32[8,8]) tuple(%[[CUSTOM_CALL_0:.*]], %[[CUSTOM_CALL_1:.*]])
        // CHECK-SAME{LITERAL}: sharding={{devices=[8,4]<=[32]}, {replicated}}
      }
    }

// -----

module @sdy_frontend_attributes_inlined_meshes {
      func.func @main(
        %arg0: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{\"x\"}, {}]>"}},
        %arg1: tensor<8x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{}, {\"y\"}]>"}}
      ) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
      // CHECK: %Arg_0.1 = f32[8,8] parameter(0)
      // CHECK-SAME: sharding={devices=[2,1,2]<=[4] last_tile_dim_replicate}
      // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{\"x\"}, {}]>"}
      // CHECK-NEXT: %Arg_1.1 = f32[8,8] parameter(1)
      // CHECK-SAME: sharding={devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}
      // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding<mesh<[\"x\"=2, \"y\"=2]>, [{}, {\"y\"}]>"}
        %0 = mhlo.add %arg0, %arg1
          {mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=2]>, [{\"x\"}, {\"y\"}]>]>"
          }}
          : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %add.1 = f32[8,8] add(%Arg_0.1, %Arg_1.1)
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=2]>, [{\"x\"}, {\"y\"}]>]>"}
        %1 = "mhlo.custom_call"(%0) {
          call_target_name = "xla.sdy.FuncResultSharding",
          mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{\"x\", \"y\", ?}, {\"z\"}]>]>"
          }
        } : (tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %[[CUSTOM_CALL_0:.*]] = f32[8,8] custom-call(%add.1)
        // CHECK-SAME: custom_call_target="xla.sdy.FuncResultSharding"
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{\"x\", \"y\", ?}, {\"z\"}]>]>"}
        // CHECK-NEXT: %[[RESHAPE_0:.*]] = f32[8,8] reshape(%[[CUSTOM_CALL_0]])
        // CHECK-SAME: sharding={devices=[8,4]<=[32]}
        %2 = "mhlo.custom_call"(%0) {
          call_target_name = "xla.sdy.FuncResultSharding",
          mhlo.frontend_attributes = {
            xla.sdy.sharding = "#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{}, {}]>]>"
          }
        } : (tensor<8x8xf32>) -> tensor<8x8xf32>
        // CHECK: %[[CUSTOM_CALL_1:.*]] = f32[8,8] custom-call(%add.1)
        // CHECK-SAME: custom_call_target="xla.sdy.FuncResultSharding"
        // CHECK-NOT: sharding={
        // CHECK-SAME: frontend_attributes={xla.sdy.sharding="#sdy.sharding_per_value<[<mesh<[\"x\"=2, \"y\"=4, \"z\"=4]>, [{}, {}]>]>"}
        // CHECK-NEXT: %[[RESHAPE_1:.*]] = f32[8,8] reshape(%[[CUSTOM_CALL_1]])
        // CHECK-SAME: sharding={replicated}
        return %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>
        // CHECK: ROOT %tuple.1 = (f32[8,8], f32[8,8]) tuple(%[[CUSTOM_CALL_0:.*]], %[[CUSTOM_CALL_1:.*]])
        // CHECK-SAME{LITERAL}: sharding={{devices=[8,4]<=[32]}, {replicated}}
      }
    }
