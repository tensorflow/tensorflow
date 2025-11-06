// RUN: ifrt-opt %s -mpmd-lower-to-ifrt -ifrt-merge-reshards -split-input-file 2>&1 | FileCheck %s

// Reshards out of Call Op are merged.

!m1_4x8 = !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"mesh2", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func public @main
func.func public @main(%arg0: !m1_4x8, %arg1: !m1_4x8)
  -> (!m2_4x8, !m2_4x8) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>,
    xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// Note that we batch the reshards out of the first call, but not the reshards
// into the second call.
// CHECK-NEXT: %[[M1_C0:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0, %arg1)
// CHECK-NEXT: %[[M1_C1:.*]]:2, %{{.*}} = ifrt.Call @f(%arg0, %arg1)
// CHECK-NEXT: %[[BATCHED_RESHARD:.*]]:2, %{{.*}} = ifrt.Reshard(%[[M1_C0]]#0, %[[M1_C0]]#1)
// CHECK-NEXT: %[[RESHARD1:.*]], %{{.*}} = ifrt.Reshard(%[[M1_C1]]#0)
// CHECK-NEXT: %[[M2_C0:.*]]:2, %{{.*}} = ifrt.Call @f(%[[BATCHED_RESHARD]]#0, %[[BATCHED_RESHARD]]#1)
// CHECK-NEXT: %[[M2_C1:.*]]:2, %{{.*}} = ifrt.Call @f(%[[BATCHED_RESHARD]]#1, %[[RESHARD1]])
// CHECK-NEXT: return %[[M2_C0]]#0, %[[M2_C1]]#1
  %m1_c0:2 = mpmd.fragment_call<mesh="mesh1", origin=[]> @f(%arg0, %arg1) : (!m1_4x8, !m1_4x8) -> (!m1_4x8, !m1_4x8)
  %m1_c1:2 = mpmd.fragment_call<mesh="mesh1", origin=[]> @f(%arg0, %arg1) : (!m1_4x8, !m1_4x8) -> (!m1_4x8, !m1_4x8)

  %t0 = mpmd.transfer %m1_c0#0 : (!m1_4x8) -> !m2_4x8
  %t1 = mpmd.transfer %m1_c0#1 : (!m1_4x8) -> !m2_4x8
  %t2 = mpmd.transfer %m1_c1#0 : (!m1_4x8) -> !m2_4x8

  %m2_c0:2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @f(%t0, %t1) : (!m2_4x8, !m2_4x8) -> (!m2_4x8, !m2_4x8)
  %m2_c1:2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @f(%t1, %t2) : (!m2_4x8, !m2_4x8) -> (!m2_4x8, !m2_4x8)
  return %m2_c0#0, %m2_c1#1 : !m2_4x8, !m2_4x8
}

func.func @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
    return %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// Reshards out of Func Args are merged.

!m1_4x8 = !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"mesh2", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func public @main
func.func public @main(%arg0: !m1_4x8, %arg1: !m1_4x8)
  -> (!m2_4x8, !m2_4x8) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>>,
    xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT: %[[BATCHED_RESHARD:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
// CHECK-NEXT: %[[M2_C0:.*]]:2, %{{.*}} = ifrt.Call @f(%[[BATCHED_RESHARD]]#0, %[[BATCHED_RESHARD]]#1)
// CHECK-NEXT: return %[[M2_C0]]#0, %[[M2_C0]]#1

  %t0 = mpmd.transfer %arg0 : (!m1_4x8) -> !m2_4x8
  %t1 = mpmd.transfer %arg1 : (!m1_4x8) -> !m2_4x8

  %m2_c0:2 = mpmd.fragment_call<mesh="mesh2", origin=[]> @f(%t0, %t1) : (!m2_4x8, !m2_4x8) -> (!m2_4x8, !m2_4x8)
  return %m2_c0#0, %m2_c0#1 : !m2_4x8, !m2_4x8
}

func.func @f(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {mesh_shape = #sdy.mesh<["x"=4]>} {
    return %arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// Reshards into ReturnOp are grouped by destination

!m1_4x8 = !mpmd.mesh_tensor<"mesh1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"mesh2", tensor<4x8xf32>>
!m3_4x8 = !mpmd.mesh_tensor<"mesh3", tensor<4x8xf32>>

// CHECK-LABEL: module
// CHECK: func.func public @main
func.func public @main(%arg0: !m1_4x8, %arg1: !m1_4x8, %arg2: !m1_4x8)
  -> (!m2_4x8, !m2_4x8, !m3_4x8) attributes {
    topology = #mpmd.topology<<"mesh1" : <["x"=4]>>, <"mesh2" : <["x"=4]>>, <"mesh3" : <["x"=4]>>>,
    xla_tpu_user_reserved_hbm_bytes = 256 : i64} {
// CHECK-NEXT: %[[BATCHED_RESHARD:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
// CHECK-NEXT: %[[RESHARD1:.*]], %{{.*}} = ifrt.Reshard(%arg2)
// CHECK-NEXT: return %[[BATCHED_RESHARD]]#0, %[[BATCHED_RESHARD]]#1, %[[RESHARD1]]

  %t0 = mpmd.transfer %arg0 : (!m1_4x8) -> !m2_4x8
  %t1 = mpmd.transfer %arg1 : (!m1_4x8) -> !m2_4x8
  %t2 = mpmd.transfer %arg2 : (!m1_4x8) -> !m3_4x8

  return %t0, %t1, %t2 : !m2_4x8, !m2_4x8, !m3_4x8
}
