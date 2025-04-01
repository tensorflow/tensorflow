// RUN: ifrt-opt %s -ifrt-remove-attrs-from-other-dialects -split-input-file | FileCheck %s

!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
// CHECK-LABEL: @non_ifrt_or_builtin_attributes_removed
// CHECK-NOT: mhlo
// CHECK: jax.buffer_donor
// CHECK: xla_tpu_user_reserved_hbm_bytes
module @non_ifrt_or_builtin_attributes_removed attributes {
    mhlo.num_partitions = 4 : i32,
    mhlo.num_replicas = 1 : i32,
    mpmd.fragments.global_view} {
  func.func @main(%arg0: !array {mhlo.sharding = "{replicated}"} loc("w1"))
      -> (!array {mhlo.other_info = ""}) attributes {ifrt.function} {
    %0, %ctrl = ifrt.Call @add_one::@main(%arg0) on devices [0,1] {
      ifrt.mesh_name = "mesh1",
      mhlo.smuggle_attr} : (!array) -> !array
    return %0 : !array loc("sum_w1")
  }

  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {jax.buffer_donor = true})
        -> tensor<2x2xi32> attributes {
          xla_tpu_user_reserved_hbm_bytes = 0 : i64} {
      %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
