// RUN: ifrt-opt %s -mpmd-lower-to-ifrt -mpmd-ifrt-add-ctrl-dependencies 2>&1 | FileCheck %s

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
func.func public @main(%arg0: !arg_0_tensor,
                       %arg1: !arg_1_tensor,
                       %arg2: !arg_2_tensor)
  -> (!res_tensor, !tmp_tensor_mesh1) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>} {
// CHECK-NEXT: %[[CALL_0:.*]], %[[CONTROL_OUTPUT_0:.*]] = ifrt.Call @stage1(%arg0, %arg2)   on devices [0, 1, 2, 3]  {ifrt.mesh_name = "mesh1"} : (!ifrt.array<tensor<3x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>, !ifrt.array<tensor<10x3xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) -> !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>
// CHECK-NEXT: %[[RESHARD:.*]], %{{.+}} = ifrt.Reshard(%[[CALL_0]]) : (!ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) -> !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>
// CHECK-NEXT: %[[CALL_1:.*]], %[[CONTROL_OUTPUT_1:.*]] = ifrt.Call @stage2(%arg1, %[[RESHARD]])   on devices [4, 5, 6, 7]  {ifrt.mesh_name = "mesh2"} : (!ifrt.array<tensor<5x7xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>, !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>) -> !ifrt.array<tensor<10x7xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>
// CHECK-NEXT: %[[CALL_2:.*]], %[[CONTROL_OUTPUT_2:.*]] = ifrt.Call @stage1(%arg0, %arg2)   after %[[CONTROL_OUTPUT_0]] on devices [0, 1, 2, 3] {ifrt.mesh_name = "mesh1"} : (!ifrt.array<tensor<3x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>, !ifrt.array<tensor<10x3xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>) -> !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>
// CHECK-NEXT: return %[[CALL_1]], %[[CALL_2]] : !ifrt.array<tensor<10x7xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [4, 5, 6, 7]>, !ifrt.array<tensor<10x5xf32>, #ifrt.sharding_param<1x1 to [0] on 4>, [0, 1, 2, 3]>

  %0 = mpmd.fragment_call<mesh="mesh1", origin=[]> @stage1(%arg0, %arg2) : (!arg_0_tensor, !arg_2_tensor) -> !tmp_tensor_mesh1
  %1 = mpmd.transfer %0 : (!tmp_tensor_mesh1) -> !tmp_tensor_mesh2
  %2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @stage2(%arg1, %1) : (!arg_1_tensor, !tmp_tensor_mesh2) -> !res_tensor
  %3 = mpmd.fragment_call<mesh="mesh1", origin=[]> @stage1(%arg0, %arg2) : (!arg_0_tensor, !arg_2_tensor) -> !tmp_tensor_mesh1
  return %2, %3 : !res_tensor, !tmp_tensor_mesh1
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
