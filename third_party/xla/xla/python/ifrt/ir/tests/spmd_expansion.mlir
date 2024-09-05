// RUN: ifrt-opt %s -spmd-expansion -split-input-file -verify-diagnostics | FileCheck %s

#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
// CHECK-LABEL: @identity_axis0_sharded
module @identity_axis0_sharded attributes {ifrt.num_devices = 2} {
  // CHECK-NEXT: func.func @main
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-NEXT: return %[[ARG]]
  // CHECK-SAME: tensor<1x2xi32>
  func.func @main(
      %arg0: tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<1x2 to [0] on 2>
// CHECK-LABEL: @identity_axis1_sharded
module @identity_axis1_sharded
    attributes {ifrt.num_devices = 2, ifrt.entry_function = "entry_func"} {
  // CHECK-NEXT: func.func @entry_func
  // CHECK-SAME: %[[ARG:.*]]: tensor<2x1xi32>
  // CHECK-NEXT: return %[[ARG]]
  // CHECK-SAME: tensor<2x1xi32>
  func.func @entry_func(
      %arg0: tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

#device = #ifrt<devices[0,1,2,3,4,5]>
#sharding = #ifrt.sharding_param<3x2 to [1,0] on 2x3>
// CHECK-LABEL: @identify_both_axes_sharded
module @identify_both_axes_sharded attributes {ifrt.num_devices = 6} {
  // CHECK-NEXT: func.func @main
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x1xi32>
  // CHECK-NEXT: return %[[ARG]]
  // CHECK-SAME: tensor<1x1xi32>
  func.func @main(
      %arg0: tensor<3x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<3x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<3x2xi32>
  }
}

// -----

#device = #ifrt<devices[0,1]>
// CHECK-LABEL: @with_func_call
module @with_func_call attributes {ifrt.num_devices = 2} {
  // CHECK-NEXT: func.func @main
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-SAME: tensor<1x2xi32>
  // CHECK: call @identify
  // CHECK-SAME: %[[ARG]]
  // CHECK-SAME: (tensor<1x2xi32>) -> tensor<1x2xi32>
  // CHECK: return
  // CHECK-SAME: tensor<1x2xi32>
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.devices = #device})
      -> (tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.devices = #device}) {
    %0 = func.call @identify(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @identify
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-NEXT: return
  // CHECK-SAME: tensor<1x2xi32>
  func.func private @identify(%arg0: tensor<2x2xi32>)
      -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

#device = #ifrt<devices[0,1]>
// CHECK-LABEL: @with_nested_func_call
module @with_nested_func_call attributes {ifrt.num_devices = 2} {
  // CHECK-NEXT: func.func @main
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-SAME: tensor<1x2xi32>
  // CHECK: call @call_identify
  // CHECK-SAME: %[[ARG]]
  // CHECK-SAME: (tensor<1x2xi32>) -> tensor<1x2xi32>
  // CHECK: return
  // CHECK-SAME: tensor<1x2xi32>
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.devices = #device})
      -> (tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.devices = #device}) {
    %0 = func.call @call_identify(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @call_identify
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-NEXT: call @identify
  // CHECK-SAME: %[[ARG]]
  // CHECK-SAME: (tensor<1x2xi32>) -> tensor<1x2xi32>
  // CHECK-NEXT: return
  // CHECK-SAME: tensor<1x2xi32>
  func.func private @call_identify(%arg0: tensor<2x2xi32>)
      -> tensor<2x2xi32> {
    %0 = func.call @identify(%arg0) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  // CHECK: func.func private @identify
  // CHECK-SAME: %[[ARG:.*]]: tensor<1x2xi32>
  // CHECK-NEXT: return
  // CHECK-SAME: tensor<1x2xi32>
  func.func private @identify(%arg0: tensor<2x2xi32>)
      -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

#sharding = #ifrt.sharding_param<1x2 to [0] on 2>
// expected-error@+1 {{cannot find entry function `main`}}
module @missing_main_function
    attributes {ifrt.num_devices = 2} {
}

// -----

#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<1x2 to [0] on 2>
// expected-error@+1 {{cannot find entry function `entry_func`}}
module @missing_entry_function
    attributes {ifrt.num_devices = 2, ifrt.entry_function = "entry_func"} {
  func.func @main(
      %arg0: tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<2x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<2x2xi32>
  }
}

// -----

#device = #ifrt<devices[0,1]>
#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
module @non_divisible_global_shape attributes {ifrt.num_devices = 2} {
  // expected-error@+1 {{Global shape is not divisible by the number of shards in dimension 0. Global size: 3, number of shards: 2}}
  func.func @main(
      %arg0: tensor<3x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device})
      -> (tensor<3x2xi32> {ifrt.sharding = #sharding,
      ifrt.devices = #device}) {
    return %arg0 : tensor<3x2xi32>
  }
}
