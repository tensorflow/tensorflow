// RUN: ifrt-opt %s -ifrt-verify-device-type-consistency='platform_names=tpu,tpu,cpu,tpu,cpu,cuda,cuda' -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @good_call_multiple
#sharding = #ifrt.sharding_param<2 to [0] on 2>
module @good_call_multiple{
  func.func @main() -> !ifrt.array<tensor<2xi32>, #sharding, [5,6]>
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @const_hlo() on devices [0,3]
        : () -> !ifrt.array<tensor<2xi32>, #sharding, [0,3]>
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0,3]
        : (!ifrt.array<tensor<2xi32>, #sharding, [0,3]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [0,3]>
    %2, %ctrl_2 = ifrt.Call @const_hlo() on devices [5,6]
        : () -> !ifrt.array<tensor<2xi32>, #sharding, [5,6]>
    %3, %ctrl_3 = ifrt.Call @identity(%2) on devices [5,6]
        : (!ifrt.array<tensor<2xi32>, #sharding, [5,6]>)
        -> !ifrt.array<tensor<2xi32>, #sharding, [5,6]>
    return %3 : !ifrt.array<tensor<2xi32>, #sharding, [5,6]>
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }

  func.func private @const_hlo() -> tensor<2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

module @out_of_bound_device_id{
  func.func @main() -> !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<2x1 to [0] on 2>,
                                   [7, 2]>
      attributes {ifrt.function} {
  // expected-error @+1 {{'ifrt.Call' op cannot find mapping for logical device id 7. Mapping size: 7}}
    %0, %ctrl_0 = ifrt.Call @hlo() on devices [7,2]
        : () -> !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [7,2]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [7,2]>
  }

  func.func private @hlo() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

module @multiple_cpu_and_tpu{
  func.func @main() -> !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<2x1 to [0] on 2>,
                                   [0, 2]>
      attributes {ifrt.function} {
  // expected-error @+1 {{'ifrt.Call' op requires a single platform type. Expected platform: tpu. Actual platform of logical device 2: cpu}}
    %0, %ctrl_0 = ifrt.Call @hlo() on devices [0,2]
        : () -> !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,2]>
  }

  func.func private @hlo() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

module @multiple_tpu_and_gpu{
  func.func @main() -> !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<2x1 to [0] on 2>,
                                   [0, 5]>
      attributes {ifrt.function} {
  // expected-error @+1 {{'ifrt.Call' op requires a single platform type. Expected platform: tpu. Actual platform of logical device 5: cuda}}
    %0, %ctrl_0 = ifrt.Call @hlo() on devices [0,5]
        : () -> !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,5]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [0,5]>
  }

  func.func private @hlo() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

module @hlo_with_cpu_type{
  func.func @main() -> !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<2x1 to [0] on 2>,
                                   [2, 4]>
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op has platform: cpu, which is incompatible with the module type inferred from callee.}}
    %0, %ctrl_0 = ifrt.Call @hlo() on devices [2,4]
        : () -> !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [2,4]>
    return %0 : !ifrt.array<tensor<2x2xi32>,
                            #ifrt.sharding_param<2x1 to [0] on 2>, [2,4]>
  }

  func.func private @hlo() -> tensor<2x2xi32> {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
