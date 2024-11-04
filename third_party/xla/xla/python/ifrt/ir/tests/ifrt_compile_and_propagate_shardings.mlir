// RUN: ifrt-opt %s -ifrt-compile-and-propagate-shardings -split-input-file | FileCheck %s

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_to_next_call_op
module @propagate_to_next_call_op {
  func.func @main(
      %arg0: !array) -> !array_unspecified attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array) -> !array_unspecified
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_1:.+]](%[[OUT_0]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_1::@main(%0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    return %1 : !array_unspecified
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_0]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_1]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_1 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @verify_only_one_module_is_compiled
module @verify_only_one_module_is_compiled {
  func.func @main(%arg0: !array) -> (!array_unspecified, !array_unspecified)
      attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array) -> !array_unspecified
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        : (!array) -> !array_unspecified
    return %0, %1 : !array_unspecified, !array_unspecified
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_0]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_to_reshard
module @propagate_to_reshard {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array) -> !array_unspecified
    // CHECK: %[[OUT_RESHARD:.+]], %{{.+}} = ifrt.Reshard(%[[OUT]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Reshard(%0) : (!array_unspecified) -> !array
    return %1 : !array
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_to_two_call_op
module @propagate_to_two_call_op {
  func.func @main(%arg0: !array) -> (!array_unspecified, !array)
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array) -> !array_unspecified
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_1:.+]](%[[OUT_0]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_1::@main(%0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    // CHECK: %[[OUT_2:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_2:.+]](%[[OUT_0]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %2, %ctrl_2 = ifrt.Call @add_one_2::@main(%0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array
    return %1, %2 : !array_unspecified, !array
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_0]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_1]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_1 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_2]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  module @add_one_2 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_from_two_call_op
module @propagate_from_two_call_op {
  func.func @main(%arg0: !array) -> (!array_unspecified, !array)
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_0:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array) -> !array_unspecified
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_1:.+]](%[[OUT_0]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_1::@main(%0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    // CHECK: %[[OUT_2:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE_2:.+]](%[[OUT_0]], %[[OUT_1]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %2, %ctrl_2 = ifrt.Call @add_args::@main(%0, %1) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified, !array_unspecified) -> !array
    return %1, %2 : !array_unspecified, !array
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_0]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_1]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_1 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE_2]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  module @add_args attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
      -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.add %arg0, %arg1 : tensor<2x2xi32>
      return %0 : tensor<2x2xi32>
    }
  }
}

// -----

!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_to_inputs
module @propagate_to_inputs {
  func.func @main(%arg0: !array_unspecified)
      -> !array_unspecified attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    return %0 : !array_unspecified
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }

}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_from_reshard
module @propagate_from_reshard {
  func.func @main(%arg0: !array_unspecified)
      -> (!array, !array_unspecified) attributes {ifrt.function} {
    // CHECK: %[[OUT_RESHARD:.+]], %{{.+}} = ifrt.Reshard(%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array_unspecified) -> !array
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    return %0, %1 : !array, !array_unspecified
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_from_copy_arrays
module @propagate_from_copy_arrays {
  func.func @main(%arg0: !array_unspecified)
      -> (!array, !array_unspecified) attributes {ifrt.function} {
    // CHECK: %[[OUT_COPY:.+]], %{{.+}} = ifrt.CopyArrays(%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array_unspecified) -> !array
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %1, %ctrl_1 = ifrt.Call @add_one_0::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array_unspecified) -> !array_unspecified
    return %0, %1 : !array, !array_unspecified
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one_0 attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32>) -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0, 1]>
// CHECK-LABEL: @propagate_to_copy_arrays
module @propagate_to_copy_arrays {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.CallLoadedExecutable @[[CALLEE:.+]](%arg0)
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0, 1]
        {ifrt.module_type = "xla"} : (!array0) -> !array_unspecified
    // CHECK: %[[OUT_COPY:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT]])
    // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
    // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [2, 3]>
    %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array_unspecified) -> !array1
    return %1 : !array1
  }

  // CHECK: ifrt.LoadedExecutable @[[CALLEE]]
  // CHECK-SAME: on devices [0, 1]
  // CHECK-SAME: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<2x2xi32> {mhlo.sharding = "{replicated}"})
      -> (tensor<2x2xi32>) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
