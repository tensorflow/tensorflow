// RUN: ifrt-opt %s -ifrt-lower-atom-program-metadata-to-xla='compile_options={{"compile_option_overrides": {"test_override": {"executable_build_options": {"use_shardy_partitioner": true}}}}}' -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL: @arg_metadata_sdy_partitioned
module @arg_metadata_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG:    ifrt.memory_kind = "device"
  // CHECK-DAG:    mhlo.memory_kind = "device"
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<1x1 to [0] on 2>
  // CHECK-SAME: }
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.memory_kind = "device"},
      %arg1: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<1x1 to [0] on 2>}) {
    return
  }
}

// -----

// CHECK-LABEL: @arg_unspecified_sharding_sdy_partitioned
module @arg_unspecified_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: }
  // CHECK: %arg1: tensor<2x2xi32> {ifrt.sharding = #ifrt.sharding_unspecified})
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>},
      %arg1: tensor<2x2xi32> {ifrt.sharding=#ifrt.sharding_unspecified}) {
    return
  }
}

// -----

// CHECK-LABEL: @result_metadata_sdy_partitioned
module @result_metadata_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> (tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-DAG:    ifrt.memory_kind = "device"
  // CHECK-DAG:    mhlo.memory_kind = "device"
  // CHECK-SAME: }
  func.func @main()
      -> (tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
        ifrt.memory_kind = "device"}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: @result_unspecified_sharding_sdy_partitioned
module @result_unspecified_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> (tensor<2x2xi32>
  // CHECK-SAME: {
  // CHECK-NOT:    mhlo.sharding
  // CHECK-DAG:    ifrt.sharding = #ifrt.sharding_param<2x1 to [0] on 2>
  // CHECK-SAME: }
  // CHECK: tensor<2x2xi32> {ifrt.sharding = #ifrt.sharding_unspecified})
  func.func @main()
      -> (tensor<2x2xi32> {
            ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>},
          tensor<2x2xi32> {ifrt.sharding=#ifrt.sharding_unspecified}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0, %0 : tensor<2x2xi32>, tensor<2x2xi32>
  }
}

// -----

// CHECK-LABEL: @arg_missing_sharding_sdy_partitioned
module @arg_missing_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: %arg0: tensor<2x2xi32>
  // CHECK-NOT: mhlo.sharding
  func.func @main(%arg0: tensor<2x2xi32>) {
    return
  }
}

// -----

// CHECK-LABEL: @result_missing_sharding_sdy_partitioned
module @result_missing_sharding_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  // CHECK: -> tensor<2x2xi32>
  // CHECK-NOT: mhlo.sharding
  func.func @main() -> (tensor<2x2xi32>) {
     %0 = mhlo.constant dense<1> : tensor<2x2xi32>
     return %0 : tensor<2x2xi32>
  }
}

// -----

// expected-error @+1 {{'builtin.module' op module `module_missing_devices_sdy_partitioned` must have `ifrt.num_devices` attribute}}
module @module_missing_devices_sdy_partitioned attributes {ifrt.compile_options_key = "test_override"} {
  func.func @main() -> (tensor<2x2xi32>
     {ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>,
       ifrt.devices=#ifrt<devices[1]>}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

// expected-error @+2 {{'func.func' op can't lower sharding of input #0. Sharding: #ifrt.sharding_param<1x1 to [0] on 1> uses 1 devices while computation uses 2 devices}}
module @arg_w_different_num_devices_sdy_partitioned attributes {ifrt.num_devices = 2, ifrt.compile_options_key = "test_override"} {
  func.func @main(
      %arg0: tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<1x1 to [0] on 1>}) {
    return
  }
}

// -----

// expected-error @+2 {{'func.func' op can't lower sharding of output #0. Sharding: #ifrt.sharding_param<2x1 to [0] on 2> uses 2 devices while computation uses 4 devices}}
module @res_w_different_num_devices_sdy_partitioned attributes {ifrt.num_devices = 4, ifrt.compile_options_key = "test_override"} {
  func.func @main()
      -> (tensor<2x2xi32> {
        ifrt.sharding=#ifrt.sharding_param<2x1 to [0] on 2>}) {
    %0 = mhlo.constant dense<1> : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}