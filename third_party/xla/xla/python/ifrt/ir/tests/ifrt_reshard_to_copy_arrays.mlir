// RUN: ifrt-opt %s -ifrt-reshard-to-copy-arrays -verify-diagnostics -split-input-file | FileCheck %s

!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
// CHECK-LABEL: @reshard_to_copy_arrays
module @reshard_to_copy_arrays {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[COPIED:.+]], %{{.+}} = ifrt.CopyArrays(%arg0)
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: return %[[COPIED]]
    return %0 : !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [2,3]>
// CHECK-LABEL: @reshard_not_converted
module @reshard_not_converted {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[RESHARDED:.+]], %{{.+}} = ifrt.Reshard(%arg0)
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: return %[[RESHARDED]]
    return %0 : !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [2,3]>
// CHECK-LABEL: @extract_copy_from_reshard
module @extract_copy_from_reshard {
  func.func @main(%arg0: !array0, %arg1: !array1) -> (!array1, !array2)
      attributes {ifrt.function} {
    // CHECK: %[[RESHARDED:.+]], %{{.+}} = ifrt.Reshard(%arg1) {donated = true}
    // CHECK: %[[COPIED:.+]], %{{.+}} = ifrt.CopyArrays(%arg0) {donated = true}
    %0, %1, %ctrl_0 = ifrt.Reshard(%arg0, %arg1) {donated = true}
        : (!array0, !array1) -> (!array1, !array2)
    // CHECK: return %[[COPIED]], %[[RESHARDED]]
    return %0, %1: !array1, !array2
  }
}

// -----

// Verifies that an ifrt.CopyArrays is introduced for each set of devices.
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
// CHECK-LABEL: @extract_copy_per_device_set
module @extract_copy_per_device_set {
  func.func @main(%arg0: !array0, %arg1: !array1, %arg2: !array1)
      -> (!array1, !array2, !array0) attributes {ifrt.function} {
    // CHECK: %[[RESHARDED:.+]], %{{.+}} = ifrt.Reshard(%arg1)
    // CHECK-DAG: %[[COPIED_1:.+]], %{{.+}} = ifrt.CopyArrays(%arg0)
    // CHECK-DAG: %[[COPIED_2:.+]], %{{.+}} = ifrt.CopyArrays(%arg2)
    %0, %1, %2, %ctrl_0 = ifrt.Reshard(%arg0, %arg1, %arg2)
        : (!array0, !array1, !array1) -> (!array1, !array2, !array0)
    // CHECK: return %[[COPIED_1]], %[[RESHARDED]], %[[COPIED_2]]
    return %0, %1, %2: !array1, !array2, !array0
  }
}

// -----

// Verifies that the control inputs are passed to the CopyArrays.
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [2,3]>
// CHECK-LABEL: @control_inputs_added_to_copy_arrays
module @control_inputs_added_to_copy_arrays {
  func.func @main(%arg0: !array0, %arg1: !array1) -> (!array1, !array2)
      attributes {ifrt.function} {
  // CHECK: %[[OUT:.+]], %[[CTRL:.+]] = ifrt.Call @add_one(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array0) -> !array0
    // CHECK: %[[RESHARDED:.+]], %{{.+}} = ifrt.Reshard(%arg1) after %[[CTRL:.+]]
    // CHECK: %[[COPIED:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT:.+]]) after %[[CTRL:.+]]
    %1, %2, %ctrl_1 = ifrt.Reshard(%0, %arg1) after %ctrl_0
        : (!array0, !array1) -> (!array1, !array2)
    // CHECK: return %[[COPIED]], %[[RESHARDED]]
    return %1, %2: !array1, !array2
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
