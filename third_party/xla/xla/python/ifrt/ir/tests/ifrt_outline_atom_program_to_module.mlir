// RUN: ifrt-opt %s -ifrt-outline-atom-program-to-module -split-input-file -verify-diagnostics | FileCheck %s

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @call_hlo
module @call_hlo {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        {ifrt.compile_options_key = "fake_compile_options_key"}
        : (!array) -> !array
    return %0 : !array
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @calls_share_a_module
module @calls_share_a_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUTPUT:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[MODULE:.+]]::@main(%[[OUTPUT]])
    %1, %ctrl_1 = ifrt.Call @add_one(%0) on devices [0,1] : (!array) -> !array
    return %1 : !array
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}


// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @calls_with_ctrl_dep_share_a_module
module @calls_with_ctrl_dep_share_a_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUTPUT:.+]], %[[CTRL_0:.+]] = ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[MODULE:.+]]::@main(%[[OUTPUT]]) after %[[CTRL_0]]
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0,1]>
// CHECK-LABEL: @call_with_diff_sharding_share_a_module
module @call_with_diff_sharding_share_a_module {
  func.func @main(%arg0: !array) -> !array_unspecified
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0, 1]
        : (!array) -> !array
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%[[OUT_0]])
    %1, %ctrl_1 = ifrt.Call @add_one(%0) on devices [0, 1]
        : (!array) -> !array_unspecified
    // CHECK: return %[[OUT_1]]
    return %1 : !array_unspecified
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>

// CHECK-LABEL: @call_with_diff_devices_share_a_module
module @call_with_diff_devices_share_a_module {
  func.func @main(%arg0: !array0, %arg1: !array1) -> (!array0, !array1)
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0, 1]
        : (!array0) -> !array0
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%arg1)
    %1, %ctrl_1 = ifrt.Call @add_one(%arg1) on devices [2, 3]
        : (!array1) -> !array1
    // CHECK: return %[[OUT_0]], %[[OUT_1]]
    return %0, %1 : !array0, !array1
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @shared_func_is_cloned
module @shared_func_is_cloned {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.Call @[[MODULE1:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[MODULE2:.+]]::@main(%[[OUT]])
    %1, %ctrl_1 = ifrt.Call @add_two(%0) on devices [0,1] : (!array) -> !array
    return %1 : !array
  }

  func.func private @add_one_internal(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }

  // CHECK: module @[[MODULE1]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  // CHECK: func.func private @add_one_internal
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = func.call @add_one_internal(%arg0) : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    return %0 : tensor<2x2xi32>
  }

  // CHECK: module @[[MODULE2]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  // CHECK: func.func private @add_one_internal
  func.func private @add_two(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = func.call @add_one_internal(%arg0) : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    %1 = func.call @add_one_internal(%0) : (tensor<2x2xi32>) -> (tensor<2x2xi32>)
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
// CHECK-LABEL: @callee_with_symbol
module @callee_with_symbol {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[MODULE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [2]
        : (!array) -> !array
    return %0 : !array
  }

  // CHECK: module @[[MODULE]]
  // CHECK: attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<2> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 {attr_sym = @add_two}: tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @add_two
  // CHECK-NEXT: mhlo.constant
  // CHECK-NEXT: mhlo.add
  func.func private @add_two(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<2> : tensor<2x2xi32>
    %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
module @unknown_symbol_in_callee {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [2] : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    // expected-error @+1 {{'mhlo.add' op uses a symbol in attributes `unknown` that does not exist in the ModuleOp}}
    %1 = mhlo.add %arg0, %0 {f = @unknown} : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [2]>
module @wrong_type_for_symbol_in_callee {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [2] : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    // expected-error @+1 {{'mhlo.add' op uses a symbol in attributes `a_module` that is not a FuncOp. Cannot handle such cases for now}}
    %1 = mhlo.add %arg0, %0 {f = @a_module} : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }

  module @a_module {}
}
