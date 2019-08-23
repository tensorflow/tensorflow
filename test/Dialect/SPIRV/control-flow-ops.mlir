// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

"foo.function"() ({
  // expected-error @+1 {{op must appear in a 'func' block}}
  spv.Return
})  : () -> ()

// -----

// Return mismatches function signature
spv.module "Logical" "VulkanKHR" {
  func @work() -> (i32) {
    // expected-error @+1 {{cannot be used in functions returning value}}
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

func @ret_val() -> (i32) {
  %0 = spv.constant 42 : i32
  // CHECK: spv.ReturnValue %{{.*}} : i32
  spv.ReturnValue %0 : i32
}

// -----

"foo.function"() ({
  %0 = spv.constant true
  // expected-error @+1 {{op must appear in a 'func' block}}
  spv.ReturnValue %0 : i1
})  : () -> ()

// -----

func @value_count_mismatch() -> () {
  %0 = spv.constant 42 : i32
  // expected-error @+1 {{op returns 1 value but enclosing function requires 0 results}}
  spv.ReturnValue %0 : i32
}

// -----

func @value_type_mismatch() -> (f32) {
  %0 = spv.constant 42 : i32
  // expected-error @+1 {{return value's type ('i32') mismatch with function's result type ('f32')}}
  spv.ReturnValue %0 : i32
}
