// RUN: ifrt-opt %s -ifrt-precompile-atom-program-preprocessing='platform_names=tpu,tpu compile_options={{"compile_option_overrides": {"test_override": {"executable_build_options": {"use_shardy_partitioner": true}}}}}' -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-SDY
// RUN: ifrt-opt %s -ifrt-precompile-atom-program-preprocessing='platform_names=tpu,tpu compile_options={{"compile_option_overrides": {"test_override": {"executable_build_options": {"use_shardy_partitioner": false}}}}}' -split-input-file -verify-diagnostics | FileCheck %s --check-prefixes=CHECK,CHECK-MHLO

#sharding = #ifrt.sharding_param<2x1 to [0] on 2>
!array = !ifrt.array<tensor<2x2xi32>, #sharding, [0, 1]>
  // CHECK-LABEL: @call_twice
module @call_twice {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: %[[OUT:.+]], %{{.+}} = ifrt.Call @[[MODULE:.+]]::@main(%arg0) on devices [0, 1] {ifrt.module_type = "xla"}
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0, 1]
        : (!array) -> !array
    // CHECK: ifrt.Call @[[MODULE:.+]]::@main(%[[OUT]]) on devices [0, 1] {ifrt.module_type = "xla"}
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0, 1]
        : (!array) -> !array
    return %1 : !array
  }

  // CHECK: module @[[MODULE]] attributes {sym_visibility = "private"}
  // CHECK: func.func @main
  // CHECK:  %arg0: tensor<2x2xi32>
  // CHECK-SDY-NOT: mhlo.sharding
  // CHECK-MHLO: mhlo.sharding = "{devices=[2,1]<=[2]}"
  // CHECK: tensor<2x2xi32>
  // CHECK-SDY-NOT:  mhlo.sharding
  // CHECK-MHLO: mhlo.sharding = "{devices=[2,1]<=[2]}"}
  // CHECK-NOT: ifrt
  module @add_one attributes {
        ifrt.num_devices = 2,
        sym_visibility = "private",
        ifrt.compile_options_key = "test_override"} {
    func.func @main(%arg0: tensor<2x2xi32> {
        ifrt.sharding = #sharding, ifrt.devices = #ifrt<devices[0, 1]>
    }) -> (tensor<2x2xi32> {
        ifrt.sharding = #sharding, ifrt.devices = #ifrt<devices[0, 1]>
    }) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
