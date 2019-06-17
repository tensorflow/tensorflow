// RUN: mlir-opt -split-input-file -verify %s | FileCheck %s

// TODO(antiagainst): Remove the wrapping functions once MLIR is moved to
// be generally region-based.

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

func @const() -> () {
  // CHECK: %0 = spv.constant true : i1
  // CHECK: %1 = spv.constant 42 : i32
  // CHECK: %2 = spv.constant 5.000000e-01 : f32
  // CHECK: %3 = spv.constant dense<vector<2xi32>, [2, 3]> : vector<2xi32>
  // CHECK: %4 = spv.constant [dense<vector<2xf32>, 3.000000e+00>] : !spv.array<1 x vector<2xf32>>

  %0 = spv.constant true
  %1 = spv.constant 42 : i32
  %2 = spv.constant 0.5 : f32
  %3 = spv.constant dense<vector<2xi32>, [2, 3]>
  %4 = spv.constant [dense<vector<2xf32>, 3.0>] : !spv.array<1xvector<2xf32>>
  return
}

// -----

func @unaccepted_std_attr() -> () {
  // expected-error @+1 {{cannot have value of type 'none'}}
  %0 = spv.constant unit : none
  return
}

// -----

func @array_constant() -> () {
  // expected-error @+1 {{has array element that are not of result array element type}}
  %0 = spv.constant [dense<vector<2xf32>, 3.0>, dense<vector<2xi32>, 4>] : !spv.array<2xvector<2xf32>>
  return
}

// -----

func @array_constant() -> () {
  // expected-error @+1 {{must have spv.array result type for array value}}
  %0 = spv.constant [dense<vector<2xf32>, 3.0>] : !spv.rtarray<vector<2xf32>>
  return
}

// -----

func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{result type ('vector<4xi32>') does not match value type ('tensor<4xi32>')}}
  %0 = "spv.constant"() {value: dense<tensor<4xi32>, 0>} : () -> (vector<4xi32>)
}

// -----

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @module_without_cap_ext
func @module_without_cap_ext() -> () {
  // CHECK: spv.module
  spv.module { } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// CHECK-LABEL: func @module_with_cap_ext
func @module_with_cap_ext() -> () {
  // CHECK: spv.module
  spv.module { } attributes {
    capability: ["Shader"],
    extension: ["SPV_KHR_16bit_storage"],
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// CHECK-LABEL: func @module_with_explict_module_end
func @module_with_explict_module_end() -> () {
  // CHECK: spv.module
  spv.module {
    spv._module_end
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// CHECK-LABEL: func @module_with_func
func @module_with_func() -> () {
  // CHECK: spv.module
  spv.module {
    func @do_nothing() -> () {
      spv.Return
    }
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

func @missing_addressing_model() -> () {
  // expected-error@+1 {{requires attribute 'addressing_model'}}
  spv.module { } attributes {}
  return
}

// -----

func @wrong_addressing_model() -> () {
  // expected-error@+1 {{attribute 'addressing_model' failed to satisfy constraint}}
  spv.module { } attributes {addressing_model: "Physical", memory_model: "VulkanHKR"}
  return
}

// -----

func @missing_memory_model() -> () {
  // expected-error@+1 {{requires attribute 'memory_model'}}
  spv.module { } attributes {addressing_model: "Logical"}
  return
}

// -----

func @module_with_multiple_blocks() -> () {
  // expected-error @+1 {{failed to verify constraint: region with 1 blocks}}
  spv.module {
  ^first:
    spv.Return
  ^second:
    spv.Return
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

func @use_non_spv_op_inside_module() -> () {
  spv.module {
    // expected-error @+1 {{'spv.module' can only contain func and spv.* ops}}
    "dialect.op"() : () -> ()
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

func @use_non_spv_op_inside_func() -> () {
  spv.module {
    func @do_nothing() -> () {
      // expected-error @+1 {{functions in 'spv.module' can only contain spv.* ops}}
      "dialect.op"() : () -> ()
    }
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

func @use_extern_func() -> () {
  spv.module {
    // expected-error @+1 {{'spv.module' cannot contain external functions}}
    func @extern() -> ()
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

func @module_with_nested_func() -> () {
  spv.module {
    func @outer_func() -> () {
      // expected-error @+1 {{'spv.module' cannot contain nested functions}}
      func @inner_func() -> () {
        spv.Return
      }
      spv.Return
    }
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv._module_end
//===----------------------------------------------------------------------===//

func @module_end_not_in_module() -> () {
  // expected-error @+1 {{can only be used in a 'spv.module' block}}
  spv._module_end
}
