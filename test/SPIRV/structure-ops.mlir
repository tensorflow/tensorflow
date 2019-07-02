// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// TODO(antiagainst): Remove the wrapping functions once MLIR is moved to
// be generally region-based.

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

func @const() -> () {
  // CHECK: %0 = spv.constant true : i1
  // CHECK: %1 = spv.constant 42 : i32
  // CHECK: %2 = spv.constant 5.000000e-01 : f32
  // CHECK: %3 = spv.constant dense<[2, 3]> : vector<2xi32> : vector<2xi32>
  // CHECK: %4 = spv.constant [dense<3.000000e+00> : vector<2xf32>] : !spv.array<1 x vector<2xf32>>

  %0 = spv.constant true
  %1 = spv.constant 42 : i32
  %2 = spv.constant 0.5 : f32
  %3 = spv.constant dense<[2, 3]> : vector<2xi32>
  %4 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>
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
  %0 = spv.constant [dense<3.0> : vector<2xf32>, dense<4> : vector<2xi32>] : !spv.array<2xvector<2xf32>>
  return
}

// -----

func @array_constant() -> () {
  // expected-error @+1 {{must have spv.array result type for array value}}
  %0 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.rtarray<vector<2xf32>>
  return
}

// -----

func @value_result_type_mismatch() -> () {
  // expected-error @+1 {{result type ('vector<4xi32>') does not match value type ('tensor<4xi32>')}}
  %0 = "spv.constant"() {value = dense<0> : tensor<4xi32>} : () -> (vector<4xi32>)
}

// -----

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @module_without_cap_ext
func @module_without_cap_ext() -> () {
  // CHECK: spv.module "Logical" "VulkanKHR"
  spv.module "Logical" "VulkanKHR" { }
  return
}

// CHECK-LABEL: func @module_with_cap_ext
func @module_with_cap_ext() -> () {
  // CHECK: attributes {capability = ["Shader"], extension = ["SPV_KHR_16bit_storage"]}
  spv.module "Logical" "VulkanKHR" { } attributes {
    capability = ["Shader"],
    extension = ["SPV_KHR_16bit_storage"]
  }
  return
}

// CHECK-LABEL: func @module_with_explict_module_end
func @module_with_explict_module_end() -> () {
  // CHECK: spv.module
  spv.module "Logical" "VulkanKHR" {
    spv._module_end
  }
  return
}

// CHECK-LABEL: func @module_with_func
func @module_with_func() -> () {
  // CHECK: spv.module
  spv.module "Logical" "VulkanKHR" {
    func @do_nothing() -> () {
      spv.Return
    }
  }
  return
}

// -----

func @missing_addressing_model() -> () {
  // expected-error@+1 {{requires string for addressing model}}
  spv.module { }
  return
}

// -----

func @wrong_addressing_model() -> () {
  // expected-error@+1 {{unknown addressing model: "Physical"}}
  spv.module "Physical" { }
  return
}

// -----

func @missing_memory_model() -> () {
  // expected-error@+1 {{requires string for memory model}}
  spv.module "Logical" { }
  return
}

// -----

func @wrong_memory_model() -> () {
  // expected-error@+1 {{unknown memory model: "Bla"}}
  spv.module "Logical" "Bla" { }
  return
}

// -----

func @module_with_multiple_blocks() -> () {
  // expected-error @+1 {{failed to verify constraint: region with 1 blocks}}
  spv.module "Logical" "VulkanKHR" {
  ^first:
    spv.Return
  ^second:
    spv.Return
  }
  return
}

// -----

func @use_non_spv_op_inside_module() -> () {
  spv.module "Logical" "VulkanKHR" {
    // expected-error @+1 {{'spv.module' can only contain func and spv.* ops}}
    "dialect.op"() : () -> ()
  }
  return
}

// -----

func @use_non_spv_op_inside_func() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @do_nothing() -> () {
      // expected-error @+1 {{functions in 'spv.module' can only contain spv.* ops}}
      "dialect.op"() : () -> ()
    }
  }
  return
}

// -----

func @use_extern_func() -> () {
  spv.module "Logical" "VulkanKHR" {
    // expected-error @+1 {{'spv.module' cannot contain external functions}}
    func @extern() -> ()
  }
  return
}

// -----

func @module_with_nested_func() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @outer_func() -> () {
      // expected-error @+1 {{'spv.module' cannot contain nested functions}}
      func @inner_func() -> () {
        spv.Return
      }
      spv.Return
    }
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
