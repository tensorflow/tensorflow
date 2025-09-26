// RUN: ifrt-opt %s -ifrt-to-outlined-atom-programs-pipeline -ifrt-populate-atom-program-metadata-pipeline -ifrt-outlined-atom-programs-to-compiled-pipeline='platform_names=tpu,tpu,tpu,tpu,host' -split-input-file | FileCheck %s

// Verify that the pass does not crash for a simple graph of 3 computations
// executing on two device sets.
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [2,3]>
// CHECK-LABEL: @call_hlo_different_meshes
module @call_hlo_different_meshes {
  func.func @main(%arg0: !array1, %arg1: !array2) -> (!array1, !array2)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array1) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [2,3]
        : (!array2) -> !array2
    %2, %ctrl_2 = ifrt.Call @add_one(%1) on devices [2,3]
        : (!array2) -> !array2
    return %0, %2 : !array1, !array2
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

// Verify that the pass fails if it encounters a MPMD reshard. The Reshard op
// should have already been converted to a CallOp.
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @reshard_mpmd {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Reshard' Dot graphs can be generated only after `ReshardOp`s have been converted to `CallOp`s"}}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}

// -----

// Verify that the pass does not fail if it encounters a CopyArrays op.
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
// CHECK-LABEL: @copy_arrays
module @copy_arrays {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}
