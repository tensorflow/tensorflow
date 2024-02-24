// RUN: xla-translate -verify-diagnostics -split-input-file -mlir-hlo-to-hlo %s | FileCheck %s
// RUN: xla-translate -verify-diagnostics -split-input-file -mlir-hlo-to-hlo --via-builder=true %s | FileCheck %s

module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0], offset = 0> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}
// CHECK-LABEL: hlo_module       {
// CHECK: cross_program_prefetches {
// CHECK-NEXT:    parameter: 1
// CHECK-NEXT:    index: 0
// CHECK-NEXT:  }

// -----

// expected-error@+1 {{cross_program_prefetch: parameter 2 out of range. main has only 2 arguments}}
module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 2, indices = [0], offset = 0> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}

// -----

// expected-error@+1 {{cross_program_prefetch: no subshape at given index: 0, 1}}
module attributes { mhlo.cross_program_prefetches = [ #mhlo.cross_program_prefetch<parameter = 1, indices = [0,1], offset = 0> ] } {
  func.func @copy(%arg0 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> attributes {execution_thread = "main"} {
    %0 = "mhlo.copy"(%arg0) {is_cross_program_prefetch} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %0 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
  func.func @main(%arg0 : tensor<i32>, %arg1 : tuple<tensor<2x3xi32>, tensor<i32>>) -> tuple<tensor<2x3xi32>, tensor<i32>> {
    %1 = "mhlo.async_start"(%arg1) {called_computation=@copy, execution_thread="main"} : (tuple<tensor<2x3xi32>, tensor<i32>>) -> (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>)
    %2 = "mhlo.async_done"(%1) {called_computation=@copy, execution_thread="main"} : (!mhlo.async_bundle<tuple<tuple<tensor<2x3xi32>, tensor<i32>>>, tuple<tuple<tensor<2x3xi32>, tensor<i32>>>>) -> (tuple<tensor<2x3xi32>, tensor<i32>>)
    return %2 : tuple<tensor<2x3xi32>, tensor<i32>>
  }
}


// -----

module attributes {
  mhlo.use_auto_spmd_partitioning = true,
  mhlo.is_dynamic = true } {
  func.func @main(%a : tensor<i32>, %b : tensor<?xf32, #mhlo.type_extensions<bounds = [2]>>) -> () {
    func.return
  }
}

// CHECK-LABEL: hlo_module       {
// CHECK: is_dynamic: true
// CHECK: use_auto_spmd_partitioning: true

// -----

module attributes { mhlo.spmd_output_sharding = "\08\03\1A\02\01\02\22\02\00\01", mhlo.spmd_parameters_shardings = ["\08\03\1A\02\01\02\22\02\00\01"]} {
  func.func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = "mhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
    func.return %0 : tensor<16x16xf32>
  }
}

// CHECK: spmd_output_sharding {
// CHECK:   type: OTHER
// CHECK:   tile_assignment_dimensions: 1
// CHECK:   tile_assignment_dimensions: 2
// CHECK:   tile_assignment_devices: 0
// CHECK:   tile_assignment_devices: 1
// CHECK: }
// CHECK: spmd_parameters_shardings {
// CHECK:   type: OTHER
// CHECK:   tile_assignment_dimensions: 1
// CHECK:   tile_assignment_dimensions: 2
// CHECK:   tile_assignment_devices: 0
// CHECK:   tile_assignment_devices: 1
// CHECK: }

// -----

//         CHECK-LABEL: ModuleWithFrontendAttributes
module @ModuleWithFrontendAttributes attributes {
//      CHECK{LITERAL}: frontend_attributes {
//      CHECK{LITERAL}: key: "attr_name"
//      CHECK{LITERAL}: value: "attr_value"
  mhlo.frontend_attributes = { attr_name="attr_value" }
} {
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    func.return %arg0 : tensor<1xf32>
  }
}
