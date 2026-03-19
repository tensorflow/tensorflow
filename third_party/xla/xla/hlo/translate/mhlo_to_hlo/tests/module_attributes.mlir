// RUN: xla-translate -verify-diagnostics -split-input-file -mlir-hlo-to-hlo --hlo-import-all-computations %s | FileCheck %s
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

// -----

// CHECK-LABEL: input_output_alias_module
module @input_output_alias_module attributes {
//      CHECK:   input_output_alias {
// CHECK-NEXT:    entries {
// CHECK-NEXT:      output_shape_index: 0
// CHECK-NEXT:      kind: MAY_ALIAS
// CHECK-NEXT:    }
// CHECK-NEXT:    entries {
// CHECK-NEXT:      output_shape_index: 1
// CHECK-NEXT:      parameter_number: 1
// CHECK-NEXT:      kind: MAY_ALIAS
// CHECK-NEXT:    }
// CHECK-NEXT:  }
  mhlo.input_output_alias = [
  {
    alias = 
      {
        kind = "may_alias",
        parameter_index = array<i64>,
        parameter_number = 0 : i64
      },
    output_index = array<i64: 0>
  },
  {
    alias =
    {
      kind = "may_alias",
      parameter_index = array<i64>,
      parameter_number = 1 : i64
    },
    output_index = array<i64: 1>
  }
]
} {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32> ) -> (tensor<1xf32>, tensor<1xf32>) {
    func.return %arg0, %arg1: tensor<1xf32>, tensor<1xf32>
  }
}

// -----

// Check that function order in module does not impact HLO module entry
// computation assignment.

// CHECK-LABEL: entry_computation_with_multiple
// CHECK-LABEL: host_program_shape {
// CHECK-NEXT: parameters {
// CHECK-NEXT:   element_type: BF16
// CHECK-NEXT:   dimensions: 10
// CHECK-NEXT:   dimensions: 20
// CHECK-NEXT:   layout {
// CHECK-NEXT:     minor_to_major: 0
// CHECK-NEXT:     minor_to_major: 1
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT: }
// CHECK-NEXT: parameters {
// CHECK-NEXT:   element_type: BF16
// CHECK-NEXT:   dimensions: 20
// CHECK-NEXT:   layout {
// CHECK-NEXT:     minor_to_major: 0
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT: }
// CHECK-NEXT: result {
// CHECK-NEXT:   element_type: F32
// CHECK-NEXT:   layout {
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT: }
module @entry_computation_with_multiple attributes {
    mhlo.xla_entry_computation_parameter_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>],
    mhlo.xla_entry_computation_parameter_tiles = [[], []]
  } {
  func.func @callee(%arg0: tensor<10x20xbf16>) -> tensor<f32> {
    %cst = mhlo.constant dense<1.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
  func.func @main(%arg0: tensor<10x20xbf16> {mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"}, %arg1: tensor<20xbf16> {mhlo.sharding = "{replicated}"}) -> tensor<f32> {
    %0 = call @callee(%arg0) : (tensor<10x20xbf16>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// -----

// CHECK-LABEL: entry_computation_with_multiple_swap
// CHECK-LABEL: host_program_shape {
// CHECK-NEXT: parameters {
// CHECK-NEXT:   element_type: BF16
// CHECK-NEXT:   dimensions: 10
// CHECK-NEXT:   dimensions: 20
// CHECK-NEXT:   layout {
// CHECK-NEXT:     minor_to_major: 0
// CHECK-NEXT:     minor_to_major: 1
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT: }
// CHECK-NEXT: parameters {
// CHECK-NEXT:   element_type: BF16
// CHECK-NEXT:   dimensions: 20
// CHECK-NEXT:   layout {
// CHECK-NEXT:     minor_to_major: 0
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT:   is_dynamic_dimension: false
// CHECK-NEXT: }
// CHECK-NEXT: result {
// CHECK-NEXT:   element_type: F32
// CHECK-NEXT:   layout {
// CHECK-NEXT:     tail_padding_alignment_in_elements: 1
// CHECK-NEXT:   }
// CHECK-NEXT: }
module @entry_computation_with_multiple_swap attributes {
    mhlo.xla_entry_computation_parameter_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>],
    mhlo.xla_entry_computation_parameter_tiles = [[], []]
  } {
  func.func @main(%arg0: tensor<10x20xbf16> {mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"}, %arg1: tensor<20xbf16> {mhlo.sharding = "{replicated}"}) -> tensor<f32> {
    %0 = call @callee(%arg0) : (tensor<10x20xbf16>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  func.func @callee(%arg0: tensor<10x20xbf16>) -> tensor<f32> {
    %cst = mhlo.constant dense<1.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
}
