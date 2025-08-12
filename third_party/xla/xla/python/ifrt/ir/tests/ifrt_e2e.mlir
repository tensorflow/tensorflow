// RUN: ifrt-opt %s -ifrt-to-outlined-atom-programs-pipeline -ifrt-populate-atom-program-metadata-pipeline -ifrt-outlined-atom-programs-to-compiled-pipeline='platform_names=tpu:8' -split-input-file | FileCheck %s

// CHECK-LABEL: @call_hlo
module @call_hlo {
  func.func @main(%arg0: !ifrt.array<tensor<2x2xi32>,
                                     #ifrt.sharding_param<2x1 to [0] on 2>,
                                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
      attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @[[EXEC_NAME:.+]](%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                       [0,1]>)
        -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                       [0,1]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
  }

  // CHECK: ifrt.LoadedExecutable @[[EXEC_NAME]] on devices [0, 1]
  // CHECK: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>)
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array2 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [4,5]>
// CHECK-LABEL: @call_hlo_different_meshes
module @call_hlo_different_meshes {
  func.func @main(%arg0: !array1, %arg1: !array2) -> (!array1, !array2)
      attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @[[EXEC1:.+]](%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array1) -> !array1
    // CHECK: %[[OUT1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[EXEC2:.+]](%arg1)
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [4,5]
        : (!array2) -> !array2
    // CHECK: ifrt.CallLoadedExecutable @[[EXEC2:.+]](%[[OUT1]])
    %2, %ctrl_2 = ifrt.Call @add_one(%1) on devices [4,5]
        : (!array2) -> !array2
    return %0, %2 : !array1, !array2
  }

  // CHECK: ifrt.LoadedExecutable @[[EXEC2]] on devices [4, 5]
  // CHECK: : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [4, 5]>)
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [4, 5]>
  // CHECK: ifrt.LoadedExecutable @[[EXEC1]] on devices [0, 1]
  // CHECK: : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>)
  // CHECK: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: @copy_array
module @copy_array {
  func.func @main(%arg0: !ifrt.array<tensor<2x2xi32>,
                                     #ifrt.sharding_param<2x1 to [0] on 2>,
                                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [2,3]>
      attributes {ifrt.function} {
    // CHECK: %[[COPIED:.+]], %{{.+}} = ifrt.CopyArrays(%arg0)
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0)
        : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                       [0,1]>)
        -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                       [2,3]>
    // CHECK: return %[[COPIED]]
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                      [4,5]>
// CHECK-LABEL: @reshard
// Verifies that an ifrt.Reshard is converted to a ifrt.CallLoadedExecutable.
module @reshard {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[RESHARDED:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[RESHARD_EXEC:.+]](%arg0)
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    // CHECK: return %[[RESHARDED]]
    return %0 : !array1
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @ctrl_dep_doesnt_generate_new_executable
module @ctrl_dep_doesnt_generate_new_executable {
  func.func @main(%arg0: !array) -> (!array, !array)
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %[[CTRL_0:.+]] = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0) after %[[CTRL_0]]
    %1, %ctrl_1 = ifrt.Call @add_one(%arg0) after %ctrl_0 on devices [0,1]
        : (!array) -> !array
    // CHECK: return %[[OUT_0]], %[[OUT_1]]
    return %0, %1 : !array, !array
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_0]] on devices [0, 1]
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 2>, [0, 1]>
// CHECK-LABEL: @unused_argument
module @unused_argument {
  func.func public @main(%arg0: !array, %arg1: !array) -> (!array)
    attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[EXEC_NAME:.+]](%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0, 1]
      : (!array) -> !array
    // CHECK: return %[[OUT]]
    return %0 : !array
  }

  // CHECK: ifrt.LoadedExecutable @[[EXEC_NAME]] on devices [0, 1]
  // CHECK: : (!ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 2>, [0, 1]>)
  // CHECK: -> !ifrt.array<tensor<i32>, #ifrt.sharding_param< to [0] on 2>, [0, 1]>
  func.func @add_one(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = mhlo.constant dense<1> : tensor<i32>
    %1 = mhlo.add %arg0, %0 : tensor<i32>
    return %1 : tensor<i32>
  }
}

// -----

!arr_on_mesh_0 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
!arr_on_mesh_1 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<1x1 to [0] on 2>, [4, 5]>
!arr_on_mesh_2 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<1x1 to [0] on 2>, [6, 7]>
// CHECK-LABEL: @copy_multiple_meshes
module @copy_multiple_meshes {
  func.func public @main(%arg0: !arr_on_mesh_0,
                         %arg1: !arr_on_mesh_1,
                         %arg2: !arr_on_mesh_2,
                         %arg3: !arr_on_mesh_2)
      -> (!arr_on_mesh_0, !arr_on_mesh_1, !arr_on_mesh_2) attributes {ifrt.function} {
    // CHECK: %[[COPIED_0:.+]], %{{.+}} = ifrt.CopyArrays(%arg1)
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.CopyArrays(%arg1) : (!arr_on_mesh_1) -> !arr_on_mesh_0
    // CHECK: %[[COPIED_1:.+]]:2, %{{.+}} = ifrt.CopyArrays(%arg2, %arg3)
    // CHECK: -> (!ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>, !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>)
    %1, %2, %ctrl_1 = ifrt.CopyArrays(%arg2, %arg3)
        : (!arr_on_mesh_2, !arr_on_mesh_2) -> (!arr_on_mesh_0, !arr_on_mesh_0)
    // CHECK: %[[OUT:.+]], %[[CTRL_OUT:.+]] = ifrt.CallLoadedExecutable @[[EXEC_NAME:.+]](%arg0, %[[COPIED_0]], %[[COPIED_1]]#0, %[[COPIED_1]]#1)
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %outputs, %control_output = ifrt.Call @outer(%arg0, %0, %1, %2)
        on devices [0, 1] {ifrt.local_view}
        : (!arr_on_mesh_0, !arr_on_mesh_0, !arr_on_mesh_0, !arr_on_mesh_0)
        -> !arr_on_mesh_0
    // CHECK: %[[COPIED_3:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT]])
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [4, 5]>
    %3, %ctrl_3 = ifrt.CopyArrays(%outputs) : (!arr_on_mesh_0) -> !arr_on_mesh_1
    // CHECK: %[[COPIED_4:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT]])
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [6, 7]>
    %4, %ctrl_4 = ifrt.CopyArrays(%outputs) : (!arr_on_mesh_0) -> !arr_on_mesh_2
    // CHECK: return %[[OUT]], %[[COPIED_3]], %[[COPIED_4]]
    return %outputs, %3, %4 : !arr_on_mesh_0, !arr_on_mesh_1, !arr_on_mesh_2
  }

  // CHECK: ifrt.LoadedExecutable @[[EXEC_NAME]]
  func.func @outer(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<1x8xf32>, %arg3: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<4.000000e+00> : tensor<1x8xf32>
    %2 = mhlo.reshape %arg0 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %3 = mhlo.reshape %arg1 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %4 = mhlo.reshape %arg2 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %5 = mhlo.reshape %arg3 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %6 = "mhlo.concatenate"(%2, %3, %4, %5) <{dimension = 0 : i64}> : (tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>) -> tensor<4x1x8xf32>
    %7 = mhlo.convert %6 : (tensor<4x1x8xf32>) -> tensor<4x1x8xf32>
    %8 = mhlo.reduce(%7 init: %0) applies mhlo.add across dimensions = [0] : (tensor<4x1x8xf32>, tensor<f32>) -> tensor<1x8xf32>
    %9 = mhlo.divide %8, %1 : tensor<1x8xf32>
    return %9 : tensor<1x8xf32>
  }
}

// -----

!arr_on_mesh_0 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
!arr_on_mesh_1 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [4, 5]>
!arr_on_mesh_2 = !ifrt.array<tensor<1x8xf32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [6, 7]>
// CHECK-LABEL: @reshard_multiple_meshes
module @reshard_multiple_meshes {
  func.func public @main(%arg0: !arr_on_mesh_0,
                         %arg1: !arr_on_mesh_1,
                         %arg2: !arr_on_mesh_2,
                         %arg3: !arr_on_mesh_2)
      -> (!arr_on_mesh_0, !arr_on_mesh_1, !arr_on_mesh_2) attributes {ifrt.function} {
    // CHECK: %[[RESHARDED_0:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[RESHARD_1:.+]](%arg1)
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Reshard(%arg1) : (!arr_on_mesh_1) -> !arr_on_mesh_0
    // CHECK: %[[RESHARDED_1:.+]]:2, %{{.+}} = ifrt.CallLoadedExecutable @[[RESHARD_2:.+]](%arg2, %arg3)
    // CHECK: -> (!ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>, !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>)
    %1, %2, %ctrl_1 = ifrt.Reshard(%arg2, %arg3)
        : (!arr_on_mesh_2, !arr_on_mesh_2) -> (!arr_on_mesh_0, !arr_on_mesh_0)
    // CHECK: %[[OUT:.+]], %[[CTRL_OUT:.+]] = ifrt.CallLoadedExecutable @[[EXEC_NAME:.+]](%arg0, %[[RESHARDED_0]], %[[RESHARDED_1]]#0, %[[RESHARDED_1]]#1)
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %outputs, %control_output = ifrt.Call @outer(%arg0, %0, %1, %2)
        on devices [0, 1] {ifrt.local_view}
        : (!arr_on_mesh_0, !arr_on_mesh_0, !arr_on_mesh_0, !arr_on_mesh_0)
        -> !arr_on_mesh_0
    // CHECK: %[[RESHARDED_2:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[RESHARD_2:.+]](%[[OUT]])
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<2x1 to [0] on 2>, [4, 5]>
    %3, %ctrl_3 = ifrt.Reshard(%outputs) : (!arr_on_mesh_0) -> !arr_on_mesh_1
    // CHECK: %[[RESHARDED_3:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[RESHARD_3:.+]](%[[OUT]])
    // CHECK: -> !ifrt.array<tensor<1x8xf32>, #ifrt.sharding_param<2x1 to [0] on 2>, [6, 7]>
    %4, %ctrl_4 = ifrt.Reshard(%outputs) : (!arr_on_mesh_0) -> !arr_on_mesh_2
    // CHECK: return %[[OUT]], %[[RESHARDED_2]], %[[RESHARDED_3]]
    return %outputs, %3, %4 : !arr_on_mesh_0, !arr_on_mesh_1, !arr_on_mesh_2
  }

  // CHECK: ifrt.LoadedExecutable @[[EXEC_NAME]]
  func.func @outer(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<1x8xf32>, %arg3: tensor<1x8xf32>) -> tensor<1x8xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<4.000000e+00> : tensor<1x8xf32>
    %2 = mhlo.reshape %arg0 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %3 = mhlo.reshape %arg1 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %4 = mhlo.reshape %arg2 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %5 = mhlo.reshape %arg3 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %6 = "mhlo.concatenate"(%2, %3, %4, %5) <{dimension = 0 : i64}> : (tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>, tensor<1x1x8xf32>) -> tensor<4x1x8xf32>
    %7 = mhlo.convert %6 : (tensor<4x1x8xf32>) -> tensor<4x1x8xf32>
    %8 = mhlo.reduce(%7 init: %0) applies mhlo.add across dimensions = [0] : (tensor<4x1x8xf32>, tensor<f32>) -> tensor<1x8xf32>
    %9 = mhlo.divide %8, %1 : tensor<1x8xf32>
    return %9 : tensor<1x8xf32>
  }
}
