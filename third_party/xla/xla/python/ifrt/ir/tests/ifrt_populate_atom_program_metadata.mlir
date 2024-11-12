// RUN: ifrt-opt %s -ifrt-populate-atom-program-metadata -ifrt-duplicated-callee-elimination -symbol-dce -split-input-file | FileCheck %s

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                     memory_kind = "device">
// CHECK-LABEL: @populate_arg_metadata
module @populate_arg_metadata {
  func.func @main(%arg0: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %ctrl_0 = ifrt.Call @callee::@main(%arg0) on devices [0,1] : (!array) -> ()
    return
  }

  // CHECK: module @[[CALLEE]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG: ifrt.memory_kind = "device"
  // CHECK-NOT: ifrt
  module @callee attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) {
      return
    }
  }
}

// -----

// CHECK-LABEL: @populate_result_metadata
module @populate_result_metadata {
  func.func @main() attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE:.+]]::@main()
    %0, %ctrl_0 = ifrt.Call @callee::@main() on devices [0,1]
        : () -> (!ifrt.array<tensor<2x2xi32>,
                             #ifrt.sharding_param<2x1 to [0] on 2>, [1,0],
                             memory_kind = "device">)
    return
  }

  // CHECK: module @[[CALLEE]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG: ifrt.memory_kind = "device"
  // CHECK-NOT: ifrt
  module @callee attributes {sym_visibility = "private"} {
    func.func private @main() -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      return %0 : tensor<2x2xi32>
    }
  }
}

// -----

// Verifies that a single module is populated with metadata if the input and
// output types are the same.
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @calls_outlined_to_single_module
module @calls_outlined_to_single_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: %[[OUT_1:.+]], %[[CTRL_1:.+]] = ifrt.Call @[[CALLEE]]::@main(%[[OUT_0]])
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[CALLEE]]::@main(%[[OUT_1]]) after %[[CTRL_1]]
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) after %ctrl_1 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
  }

  // CHECK: module @[[CALLEE]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-NOT: ifrt
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

// Verifies that a single module is populated with metadata even if the
// devices are different.
!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [2,3]>
// CHECK-LABEL: @calls_on_different_devices_outlined_to_single_module
module @calls_on_different_devices_outlined_to_single_module {
  func.func @main(%arg0: !array0) -> !array1 attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[CALLEE:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array0) -> !array0
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.CopyArrays(%[[OUT_0]])
    %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array0) -> (!array1)
    // CHECK: %[[OUT_2:.+]], %[[CTRL_2:.+]] = ifrt.Call @[[CALLEE]]::@main(%[[OUT_1]])
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) on devices [2,3]
        : (!array1) -> !array1
    // CHECK: ifrt.Call @[[CALLEE]]::@main(%[[OUT_2]]) after %[[CTRL_2]]
    %3, %ctrl_3 = ifrt.Call @add_one::@main(%2) after %ctrl_2 on devices [2,3]
        : (!array1) -> !array1
    return %3 : !array1
  }

  // CHECK: module @[[CALLEE]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-NOT: ifrt
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

// CHECK-LABEL: @call_twice_with_different_sharding
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
!array_unspecified = !ifrt.array<tensor<2x2xi32>,
                                 #ifrt.sharding_unspecified, [0,1]>
module @call_twice_with_different_sharding {
  func.func @main(%arg0: !array) -> !array_unspecified
      attributes {ifrt.function} {
    // CHECK: %[[OUTPUT:.+]], %{{.+}} = ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[CALLEE_1:.+]]::@main(%[[OUTPUT]])
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array_unspecified
    return %1 : !array_unspecified
  }

  // CHECK: module @[[CALLEE_0]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>

  // CHECK: module @[[CALLEE_1]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main(%arg0: tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: -> (tensor<2x2xi32>
  // CHECK-DAG: ifrt.sharding = #ifrt.sharding_unspecified
  // CHECK-NOT: ifrt
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @populate_io_alias_and_donation
module @populate_io_alias_and_donation {
  func.func @main(%arg0: !array, %arg1: !array) attributes {ifrt.function} {
    // CHECK: ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0, %arg1)
    %0, %ctrl_0 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>], donated_input_indices=array<i32: 1>}
        : (!array, !array) -> !array
    // Verify that the module is cloned if io_aliases differ.
    // CHECK: ifrt.Call @[[CALLEE_1:.+]]::@main(%arg0, %arg1)
    %1, %ctrl_1 = ifrt.Call @callee::@main(%arg0, %arg1) on devices [0,1]
        : (!array, !array) -> !array
    return
  }

  // CHECK: module @[[CALLEE_0]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main(%arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG:     tf.aliasing_output = 0 : i32
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG:     jax.buffer_donor = true
  // CHECK-SAME: }

  // CHECK: module @[[CALLEE_1]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main(%arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-DAG:     ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: }
  module @callee attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>)
         -> tensor<2x2xi32> {
      return %arg0: tensor<2x2xi32>
    }
  }
}

// -----

!shared_array = !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @output_of_call_donated
module @output_of_call_donated {
  func.func @main(%arg0: !shared_array) -> !shared_array
        attributes {ifrt.function} {
    // CHECK: %[[OUT_0:.+]], %{{.+}} = ifrt.Call @[[CALLEE_0:.+]]::@main(%arg0) on devices [0, 1] :
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!shared_array) -> !shared_array
    // CHECK: %[[OUT_1:.+]], %{{.+}} = ifrt.Call @[[CALLEE_1:.+]]::@main(%[[OUT_0]]) on devices [0, 1] {io_aliases = [array<i32: 0, 0>]} :
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!shared_array) -> !shared_array
    return %1 : !shared_array
  }

  // CHECK: module @[[CALLEE_0]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>

  // CHECK: module @[[CALLEE_1]]
  // CHECK-SAME: attributes {
  // CHECK-DAG:    ifrt.num_devices = 2
  // CHECK-DAG:    sym_visibility = "private"
  // CHECK-SAME: }
  // CHECK: func.func private @main
  // CHECK-SAME: %arg0: tensor<2x2xi32>
  // CHECK-SAME: tf.aliasing_output = 0 : i32
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
