// RUN: ifrt-opt %s -ifrt-remove-ifrt-attrs | FileCheck %s

// CHECK-LABEL: @ifrt_attributes_are_removed
// CHECK-NOT: ifrt
module @ifrt_attributes_are_removed attributes {ifrt.num_devices = 2} {
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>})
      -> (tensor<2x2xi32> {
        ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 1>,
        ifrt.memory_kind = "device"})
      attributes {ifrt.devices = #ifrt<devices[0]>} {
    return %arg0 : tensor<2x2xi32>
  }
}
