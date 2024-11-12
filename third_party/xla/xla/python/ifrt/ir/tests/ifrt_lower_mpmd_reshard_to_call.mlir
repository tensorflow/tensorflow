// RUN: ifrt-opt %s -ifrt-lower-mpmd-reshard-to-call -split-input-file -verify-diagnostics | FileCheck %s

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @reshard_without_donation
module @reshard_without_donation {
  func.func public @main(%arg0: !array0) -> (!array1)
      attributes {ifrt.function} {
    // CHECK: ifrt.Call @reshard_4784300543980450571::@main(%arg0) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @reshard_with_donation
module @reshard_with_donation {
  func.func public @main(%arg0: !array0) -> (!array1)
      attributes {ifrt.function} {
    // CHECK: ifrt.Call @reshard_4784300543980450571::@main(%arg0) on devices [0, 1, 2]
    // CHECK-SAME: {
    // CHECK-DAG:    ifrt.module_type = "mpmd_reshard"
    // CHECK-DAG:    donated_input_indices = array<i32: 0>
    // CHECK-SAME: }
    %0, %ctrl_0 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    return %0 : !array1
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2, 3]>
// ifrt.Reshard does not need to be converted to a MPMD reshard ifrt.Call
// because the reshard is a 1:1 buffer copy between devices.
module @reshard_is_not_converted_to_call {
  func.func public @main(%arg0: !array0) -> (!array1)
      attributes {ifrt.function} {
    // expected-error@+1 {{'ifrt.Reshard' op does not reshard any arrays. Use CopyArraysOp instead}}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @reshard_after_call_to_module
module @reshard_after_call_to_module {
  func.func public @main(%arg0: !array0) -> (!array1)
      attributes {ifrt.function} {
    // CHECK: %[[OUT_1:.*]], %[[CTRL_OUT:.*]] = ifrt.Call @add_one
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0, 1]
      : (!array0) -> !array0
    // CHECK: %[[OUT_2:.*]], %{{.+}} = ifrt.Call @reshard_4784300543980450571::@main(%[[OUT_1]]) after %[[CTRL_OUT]]
    // CHECK: {ifrt.module_type = "mpmd_reshard"}
    %1, %ctrl_1 = ifrt.Reshard(%0) after %ctrl_0 : (!array0) -> !array1
    return %1 : !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @reshard_before_call_to_module
module @reshard_before_call_to_module {
  func.func public @main(%arg0: !array0) -> (!array1)
      attributes {ifrt.function} {
    // CHECK: ifrt.Call @reshard_4784300543980450571::@main(%arg0) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: %[[OUT:.*]], %[[CTRL_OUT:.*]] = ifrt.Call @add_one
    %1, %ctrl_1 = ifrt.Call @add_one(%0) on devices [2]
      : (!array1) -> !array1
    return %1 : !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @two_identical_reshards_single_module
module @two_identical_reshards_single_module {
  func.func public @main(%arg0: !array0, %arg1: !array0) -> (!array1, !array1)
      attributes {ifrt.function} {
    // CHECK: ifrt.Call @reshard_4784300543980450571::@main(%arg0) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: ifrt.Call @reshard_4784300543980450571::@main(%arg1) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %1, %ctrl_1 = ifrt.Reshard(%arg1) : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @two_reshards_two_modules
module @two_reshards_two_modules {
  func.func public @main(%arg0: !array0) -> (!array0)
      attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.Call @reshard_4784300543980450571::@main(%arg0) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: ifrt.Call @reshard_17322361279023763284::@main(%[[OUT]]) on devices [0, 1, 2] {ifrt.module_type = "mpmd_reshard"}
    %1, %ctrl_1 = ifrt.Reshard(%0) : (!array1) -> !array0
    return %1 : !array0
  }

  // CHECK: module @reshard_4784300543980450571
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>

  // CHECK: module @reshard_17322361279023763284
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 3
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
// Tests if the module for the MPMD reshard has unique devices.
// CHECK-LABEL: @check_reshard_module_has_unique_devices
module @check_reshard_module_has_unique_devices {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: ifrt.Call @reshard_6746659470058475136::@main(%arg0) on devices [0, 1] {ifrt.module_type = "mpmd_reshard"}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0 : !array1
  }

  // CHECK: module @reshard_6746659470058475136
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func @main(
  // CHECK: %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
}
