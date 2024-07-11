// RUN: ifrt-opt %s -ifrt-verify-sharding-specified -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @good_arrays
#sharding = #ifrt.sharding_param<2 to [0] on 2, memory_kind = "device">
module @good_arrays {
  func.func @main(%arg0: !ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
      -> !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
      attributes {ifrt.function} {
    %0, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
    %1 = "ifrt.Reshard"(%0)
        : (!ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
    return %1 : !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

module @main_arg_sharding_unspecified {
  // expected-error @+1 {{'func.func' op argument 0 has unspecified sharding.}}
  func.func @main(
      %arg0: !ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [0,1]>)
      attributes {ifrt.function} {
    return
  }
}

// -----

#sharding = #ifrt.sharding_param<2 to [0] on 2>
module @main_result_sharding_unspecified {
  func.func @main()
      -> !ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [0,1]>
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op result 0 has unspecified sharding.}}
    %0, %ctrl_1 = ifrt.Call @create_array() on devices [0,1]
        : () -> !ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [0,1]>
    return %0 : !ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [0,1]>
  }

  func.func private @create_array() -> tensor<2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

#sharding = #ifrt.sharding_param<2 to [0] on 2>
module @reshard_with_unspecified_sharding {
  func.func @main(%arg0: !ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
      -> !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Reshard' op result 0 has unspecified sharding.}}
    %0 = ifrt.Reshard(%arg0)
        : (!ifrt.array<tensor<2xi32>, #sharding, [0,1]>)
        -> !ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [2,3]>
    %1 = ifrt.Reshard(%0)
        : (!ifrt.array<tensor<2xi32>, #ifrt.sharding_unspecified, [2,3]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
    return %1 : !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
  }
}
