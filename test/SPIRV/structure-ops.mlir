// RUN: mlir-opt -split-input-file -verify %s | FileCheck %s

// TODO(antiagainst): Remove the wrapping functions once MLIR is moved to
// be generally region-based.

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

func @module_without_cap_ext() -> () {
  // CHECK: spv.module
  spv.module { } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}

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
