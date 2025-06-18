// RUN: ifrt-opt %s -ifrt-compile-atom-program -split-input-file | FileCheck %s

// CHECK-LABEL: @call_hlo
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @call_hlo {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @fake_component__fake_method
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        {ifrt.module_type = "xla"} : (!array) -> !array
    return %0 : !array
  }

  // CHECK: ifrt.LoadedExecutable @fake_component__fake_method
  // CHECK-SAME: on devices [0, 1]
  // CHECK: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>)
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(
        %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
        -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}

// -----

!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
// CHECK-LABEL: @call_hlo_sdy_lowered
module @call_hlo_sdy_lowered attributes {
    ifrt.sdy.meshes ="{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"} {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    // CHECK: ifrt.CallLoadedExecutable @fake_component__fake_method_1(%arg0)
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        {ifrt.module_type = "xla", ifrt.is_sdy_partitioned} : (!array) -> !array
    return %0 : !array
  }

  // module @add_one attributes {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\\\22x\\\22=2]>}"}, sym_visibility = "private"}
  // CHECK: ifrt.LoadedExecutable @fake_component__fake_method
  // CHECK-SAME: on devices [0, 1]
  // CHECK: (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>)
  // CHECK-SAME: -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1]>
  module @add_one attributes {sym_visibility = "private"} {
    func.func private @main(
        %arg0: tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"})
        -> (tensor<2x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) {
      %0 = mhlo.constant dense<1> : tensor<2x2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
