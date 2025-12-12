// RUN: ifrt-opt %s -mpmd-lower-to-ifrt -verify-diagnostics -split-input-file 2>&1 | FileCheck %s

// Check that the MeshTensors are correctly converted to IFRT Arrays.
!arg_type_0 = !mpmd.mesh_tensor<"mesh1", tensor<1x2x3xf32>>
!arg_type_1 = !mpmd.mesh_tensor<"mesh1", tensor<1xf32>>
!arg_type_2 = !mpmd.mesh_tensor<"mesh2", tensor<4x2x8xf32>, sharding=<@mesh, [{"y"}, {"x"}, {"z"}]>>
!arg_type_3 = !mpmd.mesh_tensor<"mesh2", tensor<12x2x9xf32>, sharding=<@mesh, [{"x", "y","z"}, {?}, {?}]>>
!arg_type_4 = !mpmd.mesh_tensor<"mesh2", tensor<12x2x9xf32>, sharding=<@mesh, [{"y"}, {?}, {?}]>>

// CHECK-LABEL: module

// CHECK-NOT: sdy.mesh
sdy.mesh @mesh_0 = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_1 = <["axis_0"=2, "axis_1"=4, "axis_2"=4]>

// CHECK: func.func public @main
// CHECK-SAME:   %arg0: !ifrt.array<tensor<1x2x3xf32>, #ifrt.sharding_param<1x1x1 to [0] on 8>, #devices>
// CHECK-SAME:   %arg1: !ifrt.array<tensor<1xf32>, #ifrt.sharding_param<1 to [0] on 8>, #devices>
// CHECK-SAME:   %arg2: !ifrt.array<tensor<4x2x8xf32>, #ifrt.sharding_param<2x2x2 to [0, 2, 1] on 2x2x2>, #devices1>
// CHECK-SAME:   %arg3: !ifrt.array<tensor<12x2x9xf32>, #ifrt.sharding_param<8x1x1 to [0] on 8>, #devices1>
// CHECK-SAME:   %arg4: !ifrt.array<tensor<12x2x9xf32>, #ifrt.sharding_param<2x1x1 to [0, 2, 1] on 2x2x2>, #devices1>
// CHECK-SAME:   xla_tpu_user_reserved_hbm_bytes = 128 : i64
func.func public @main(%arg0: !arg_type_0,
                       %arg1: !arg_type_1,
                       %arg2: !arg_type_2,
                       %arg3: !arg_type_3,
                       %arg4: !arg_type_4)
    -> (!arg_type_0, !arg_type_1, !arg_type_2, !arg_type_3, !arg_type_4) attributes {
    topology = #mpmd.topology<
        <"mesh1" : <["x"=2, "y"=2, "z"=2]>>, <"mesh2" : <["x"=2, "y"=2, "z"=2]>>>,
        xla_tpu_user_reserved_hbm_bytes = 128 : i64} {
      return %arg0, %arg1, %arg2, %arg3, %arg4 : !arg_type_0, !arg_type_1, !arg_type_2, !arg_type_3, !arg_type_4
}

// -----

!arg_0_tensor = !mpmd.mesh_tensor<"mesh1", tensor<3x5xf32>>
!arg_1_tensor = !mpmd.mesh_tensor<"mesh2", tensor<5x7xf32>>
!arg_2_tensor = !mpmd.mesh_tensor<"mesh1", tensor<10x3xf32>>
!tmp_tensor_mesh1 = !mpmd.mesh_tensor<"mesh1", tensor<10x5xf32>>
!tmp_tensor_mesh2 = !mpmd.mesh_tensor<"mesh2", tensor<10x5xf32>>
!res_tensor = !mpmd.mesh_tensor<"mesh2", tensor<10x7xf32>>

// CHECK-LABEL: module
// CHECK: func.func public @main
// CHECK-SAME:      %arg0: !ifrt.array<tensor<3x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>
// CHECK-SAME:      %arg1: !ifrt.array<tensor<5x7xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>
// CHECK-SAME:      %arg2: !ifrt.array<tensor<10x3xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>
// CHECK-SAME:      xla_tpu_user_reserved_hbm_bytes = 256 : i64
func.func public @main(%arg0: !arg_0_tensor,
                       %arg1: !arg_1_tensor,
                       %arg2: !arg_2_tensor)
  -> (!res_tensor) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>,
    xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
      // CHECK-NEXT: %[[OUTPUTS_0:.*]], %[[CONTROL_OUTPUT_0:.*]] = ifrt.Call @stage1(%arg0, %arg2) on devices [0, 1, 2, 3] {ifrt.mesh_name = "mesh1"} : (!ifrt.array<tensor<3x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>, !ifrt.array<tensor<10x3xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) -> !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>
      // CHECK-NEXT: %[[RESHARD:.*]], %{{.+}} = ifrt.Reshard(%[[OUTPUTS_0]]) : (!ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) -> !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>
      // CHECK-NEXT: %[[OUTPUTS_1:.*]], %[[CONTROL_OUTPUT_1:.*]] = ifrt.Call @stage2(%arg1, %[[RESHARD]])
      // CHECK-NEXT: return %[[OUTPUTS_1]]
      %0 = mpmd.fragment_call<mesh="mesh1", origin=[]> @stage1(%arg0, %arg2) {mpmd.is_gspmd_partitioned} : (!arg_0_tensor, !arg_2_tensor) -> !tmp_tensor_mesh1
      %1 = mpmd.transfer %0 : (!tmp_tensor_mesh1) -> !tmp_tensor_mesh2
      %2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @stage2(%arg1, %1) : (!arg_1_tensor, !tmp_tensor_mesh2) -> !res_tensor
      return %2 : !res_tensor
}
// CHECK: func.func @stage1(%arg0: tensor<3x5xf32>, %arg1: tensor<10x3xf32>) -> tensor<10x5xf32> {
func.func @stage1(%arg0: tensor<3x5xf32>, %arg1: tensor<10x3xf32>)
  -> tensor<10x5xf32> attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
    %0 = "stablehlo.dot"(%arg1, %arg0) : (tensor<10x3xf32>, tensor<3x5xf32>) -> tensor<10x5xf32>
    return %0 : tensor<10x5xf32>
}
// CHECK: func.func @stage2(%arg0: tensor<5x7xf32>, %arg1: tensor<10x5xf32>) -> tensor<10x7xf32> {
func.func @stage2(%arg0: tensor<5x7xf32>, %arg1: tensor<10x5xf32>)
  -> tensor<10x7xf32> attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
    %0 = "stablehlo.dot"(%arg1, %arg0) : (tensor<10x5xf32>, tensor<5x7xf32>) -> tensor<10x7xf32>
    return %0 : tensor<10x7xf32>
}

// -----

!tensor = !mpmd.mesh_tensor<"mesh1", tensor<2x2xi32>>

// CHECK-LABEL: module @aliasing_output_to_io_aliases
module @aliasing_output_to_io_aliases {
  func.func public @main(%arg0: !tensor) -> (!tensor)
      attributes {topology = #mpmd.topology<<"mesh1" : <["x"=2]>>>} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.Call @add_args(%arg0, %arg0)
    // CHECK-SAME: on devices [0, 1]
    // CHECK-SAME: {
    // CHECK-DAG:    ifrt.mesh_name = "mesh1"
    // CHECK-DAG:    io_aliases = [array<i32: 1, 0>]
    // CHECK-SAME: }
    %0 = mpmd.fragment_call<mesh="mesh1", origin=[]> @add_args(%arg0, %arg0) {mpmd.is_gspmd_partitioned} : (!tensor, !tensor) -> (!tensor)
    return %0 : !tensor
  }

  // CHECK: func.func @add_args(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32> {tf.aliasing_output = 0 : i32}) -> tensor<2x2xi32> {
  func.func @add_args(%arg0: tensor<2x2xi32>,
                      %arg1: tensor<2x2xi32> {tf.aliasing_output = 0 : i32})
      -> tensor<2x2xi32> attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

!tensor_on_host = !mpmd.mesh_tensor<"mesh1", tensor<2x2xi32>, memory_kind = "pinned_host">
!tensor_on_device = !mpmd.mesh_tensor<"mesh1", tensor<2x2xi32>, memory_kind = "device">

// CHECK-LABEL: module @fetch_from_host_to_device
module @fetch_from_host_to_device {
  func.func public @main(%arg0: !tensor_on_host) -> (!tensor_on_device)
      attributes {topology = #mpmd.topology<<"mesh1" : <["x"=2]>>>} {
    // CHECK: ifrt.Reshard(%arg0)
    // CHECK-SAME: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1], memory_kind = "pinned_host">) ->
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1], memory_kind = "device">
    %0 = mpmd.transfer %arg0 : (!tensor_on_host) -> (!tensor_on_device)
    return %0 : !tensor_on_device
  }
}

// -----

!arg0_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>,
                                        sharding=<@mesh, [{"x"}, {?}]>>
!res_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>,
                                       sharding=<@mesh, [{"x"}, {?}]>>

// CHECK-LABEL: module @sdy_lowered_fragment
// CHECK-SAME: attributes {
// CHECK-DAG:    mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"}
// CHECK-SAME: }
module @sdy_lowered_fragment attributes {
    mhlo.frontend_attributes = {
      xla.sdy.meshes ="{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"}}  {
  sdy.mesh @mesh = <["x"=2]>
  func.func public @main(%arg0: !arg0_tensor) ->  !res_tensor attributes {
      topology = #mpmd.topology<<"m1" : <["x"=2]>>>} {
    // CHECK: ifrt.Call @f(%arg0) on devices [0, 1]
    // CHECK-SAME: {
    // CHECK-DAG:    ifrt.mesh_name = "m1"
    // CHECK-SAME: }
    // CHECK-SAME: (!ifrt.array<tensor<4x8xf32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>) ->
    // CHECK-SAME: !ifrt.array<tensor<4x8xf32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %0 = mpmd.fragment_call<mesh="m1", origin=["f1"]> @"f"(%arg0) {
      mpmd.is_sdy_partitioned} : (!arg0_tensor) -> !res_tensor
    return %0 : !res_tensor
  }
  // CHECK: func.func @f(%arg0: tensor<4x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22x\\\22, ?}, {?}]>"}})
  // CHECK-SAME: -> (tensor<4x8xf32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22x\\\22, ?}, {?}]>"}})
  func.func @"f"(%arg0: tensor<4x8xf32> {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22x\\\22, ?}, {?}]>"}})
      -> (tensor<4x8xf32> {
        mhlo.frontend_attributes = {
          xla.sdy.sharding = "#sdy.sharding<@mesh, [{\\\22x\\\22, ?}, {?}]>"}})
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
    return %arg0 : tensor<4x8xf32>
  }
}

// -----

!tensor_on_mesh1 = !mpmd.mesh_tensor<"mesh1", tensor<2x2xi32>, sharding=<@mesh, [{}, {}]>>
!tensor_on_mesh2 = !mpmd.mesh_tensor<"mesh2", tensor<2x2xi32>>

// CHECK-LABEL: module @copy_from_mesh1_to_mesh2_same_shape
module @copy_from_mesh1_to_mesh2_same_shape {
  func.func public @main(%arg0: !tensor_on_mesh1) -> (!tensor_on_mesh2)
      attributes {topology = #mpmd.topology<<"mesh1" : <["x"=2]>>, <"mesh2" : <["x"=2]>>>} {
    // CHECK: ifrt.Reshard(%arg0)
    // CHECK-SAME: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>) ->
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>
    %0 = mpmd.transfer %arg0 : (!tensor_on_mesh1) -> (!tensor_on_mesh2)
    return %0 : !tensor_on_mesh2
  }
}

// -----

!tensor_on_mesh1 = !mpmd.mesh_tensor<"mesh1", tensor<2x2xi32>, sharding=<@mesh, [{}, {}]>>
!tensor_on_mesh2 = !mpmd.mesh_tensor<"mesh2", tensor<2x2xi32>>

// CHECK-LABEL: module @copy_from_mesh1_to_mesh2_different_shape
module @copy_from_mesh1_to_mesh2_different_shape {
  func.func public @main(%arg0: !tensor_on_mesh1) -> (!tensor_on_mesh2)
      attributes {topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=2]>>>} {
    // CHECK: ifrt.Reshard(%arg0)
    // CHECK-SAME: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) ->
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [4, 5]>
    %0 = mpmd.transfer %arg0 : (!tensor_on_mesh1) -> (!tensor_on_mesh2)
    return %0 : !tensor_on_mesh2
  }
}
