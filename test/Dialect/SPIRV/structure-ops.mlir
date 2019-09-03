// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv._address_of
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
  func @access_chain() -> () {
    %0 = spv.constant 1: i32
    // CHECK: [[VAR1:%.*]] = spv._address_of @var1 : !spv.ptr<!spv.struct<f32, !spv.array<4 x f32>>, Input>
    // CHECK-NEXT: spv.AccessChain [[VAR1]][{{.*}}, {{.*}}] : !spv.ptr<!spv.struct<f32, !spv.array<4 x f32>>, Input>
    %1 = spv._address_of @var1 : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
    %2 = spv.AccessChain %1[%0, %0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
    spv.Return
  }
}

// -----

spv.module "Logical" "VulkanKHR" {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
  func @foo() -> () {
    // expected-error @+1 {{expected spv.globalVariable symbol}}
    %0 = spv._address_of @var2 : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
  }
}

// -----

spv.module "Logical" "VulkanKHR" {
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Input>
  func @foo() -> () {
    // expected-error @+1 {{result type mismatch with the referenced global variable's type}}
    %0 = spv._address_of @var1 : !spv.ptr<f32, Input>
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

func @const() -> () {
  // CHECK: %0 = spv.constant true
  // CHECK: %1 = spv.constant 42 : i32
  // CHECK: %2 = spv.constant 5.000000e-01 : f32
  // CHECK: %3 = spv.constant dense<[2, 3]> : vector<2xi32>
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
  // expected-error @+1 {{has array element whose type ('vector<2xi32>') does not match the result element type ('vector<2xf32>')}}
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
// spv.EntryPoint
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_nothing
   spv.EntryPoint "GLCompute" @do_nothing
}

spv.module "Logical" "VulkanKHR" {
   spv.globalVariable @var2 : !spv.ptr<f32, Input>
   spv.globalVariable @var3 : !spv.ptr<f32, Output>
   func @do_something(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () {
     %1 = spv.Load "Input" %arg0 : f32
     spv.Store "Output" %arg1, %1 : f32
     spv.Return
   }
   // CHECK: spv.EntryPoint "GLCompute" @do_something, @var2, @var3
   spv.EntryPoint "GLCompute" @do_something, @var2, @var3
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{invalid kind of constant specified}}
   spv.EntryPoint "GLCompute" "do_nothing"
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{function 'do_something' not found in 'spv.module'}}
   spv.EntryPoint "GLCompute" @do_something
}

/// TODO(ravishankarm) : Add a test that verifies an error is thrown
/// when interface entries of EntryPointOp are not
/// spv.Variables. There is currently no other op that has a spv.ptr
/// return type

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     // expected-error @+1 {{'spv.EntryPoint' op failed to verify that op must appear in a 'spv.module' block}}
     spv.EntryPoint "GLCompute" @do_something
   }
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{duplicate of a previous EntryPointOp}}
   spv.EntryPoint "GLCompute" @do_nothing
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.EntryPoint' invalid execution_model attribute specification: "ContractionOff"}}
   spv.EntryPoint "ContractionOff" @do_nothing
}

// -----

//===----------------------------------------------------------------------===//
// spv.globalVariable
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
  // CHECK: spv.globalVariable @var0 : !spv.ptr<f32, Input>
  spv.globalVariable @var0 : !spv.ptr<f32, Input>
}

// TODO: Fix test case after initialization with normal constant is addressed
// spv.module "Logical" "VulkanKHR" {
//   %0 = spv.constant 4.0 : f32
//   // CHECK1: spv.Variable init(%0) : !spv.ptr<f32, Private>
//   spv.globalVariable @var1 init(%0) : !spv.ptr<f32, Private>
// }

// -----

spv.module "Logical" "VulkanKHR" {
  spv.specConstant @sc = 4.0 : f32
  // CHECK: spv.globalVariable @var initializer(@sc) : !spv.ptr<f32, Private>
  spv.globalVariable @var initializer(@sc) : !spv.ptr<f32, Private>
}

// -----

spv.module "Logical" "VulkanKHR" {
  // CHECK: spv.globalVariable @var0 bind(1, 2) : !spv.ptr<f32, Uniform>
  spv.globalVariable @var0 bind(1, 2) : !spv.ptr<f32, Uniform>
}

// TODO: Fix test case after initialization with constant is addressed
// spv.module "Logical" "VulkanKHR" {
//   %0 = spv.constant 4.0 : f32
//   // CHECK1: spv.globalVariable @var1 initializer(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
//   spv.globalVariable @var1 initializer(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
// }

// -----

spv.module "Logical" "VulkanKHR" {
  // CHECK: spv.globalVariable @var1 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var1 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable @var2 built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var2 {built_in = "GlobalInvocationID"} : !spv.ptr<vector<3xi32>, Input>
}

// -----

spv.module "Logical" "VulkanKHR" {
  // expected-error @+1 {{expected spv.ptr type}}
  spv.globalVariable @var0 : f32
}

// -----

spv.module "Logical" "VulkanKHR" {
  // expected-error @+1 {{op initializer must be result of a spv.specConstant or spv.globalVariable op}}
  spv.globalVariable @var0 initializer(@var1) : !spv.ptr<f32, Private>
}

// -----

spv.module "Logical" "VulkanKHR" {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  spv.globalVariable @var0 : !spv.ptr<f32, Generic>
}

// -----

spv.module "Logical" "VulkanKHR" {
  func @foo() {
    // expected-error @+1 {{op failed to verify that op must appear in a 'spv.module' block}}
    spv.globalVariable @var0 : !spv.ptr<f32, Input>
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.module
//===----------------------------------------------------------------------===//

// Module without capability and extension
// CHECK: spv.module "Logical" "VulkanKHR"
spv.module "Logical" "VulkanKHR" { }

// Module with capability and extension
// CHECK: attributes {capability = ["Shader"], extension = ["SPV_KHR_16bit_storage"]}
spv.module "Logical" "VulkanKHR" { } attributes {
  capability = ["Shader"],
  extension = ["SPV_KHR_16bit_storage"]
}

// Module with explict spv._module_end
// CHECK: spv.module
spv.module "Logical" "VulkanKHR" {
  spv._module_end
}

// Module with function
// CHECK: spv.module
spv.module "Logical" "VulkanKHR" {
  func @do_nothing() -> () {
    spv.Return
  }
}

// -----

// Missing addressing model
// expected-error@+1 {{custom op 'spv.module' expected addressing_model attribute specified as string}}
spv.module { }

// -----

// Wrong addressing model
// expected-error@+1 {{custom op 'spv.module' invalid addressing_model attribute specification: "Physical"}}
spv.module "Physical" { }

// -----

// Missing memory model
// expected-error@+1 {{custom op 'spv.module' expected memory_model attribute specified as string}}
spv.module "Logical" { }

// -----

// Wrong memory model
// expected-error@+1 {{custom op 'spv.module' invalid memory_model attribute specification: "Bla"}}
spv.module "Logical" "Bla" { }

// -----

// Module with multiple blocks
// expected-error @+1 {{expects region #0 to have 0 or 1 blocks}}
spv.module "Logical" "VulkanKHR" {
^first:
  spv.Return
^second:
  spv.Return
}

// -----

// Module with wrong terminator
// expected-error@+2 {{expects regions to end with 'spv._module_end'}}
// expected-note@+1 {{in custom textual format, the absence of terminator implies 'spv._module_end'}}
"spv.module"() ({
  %0 = spv.constant true
}) {addressing_model = 0 : i32, memory_model = 1 : i32} : () -> ()

// -----

// Use non SPIR-V op inside.module
spv.module "Logical" "VulkanKHR" {
  // expected-error @+1 {{'spv.module' can only contain func and spv.* ops}}
  "dialect.op"() : () -> ()
}

// -----

// Use non SPIR-V op inside function
spv.module "Logical" "VulkanKHR" {
  func @do_nothing() -> () {
    // expected-error @+1 {{functions in 'spv.module' can only contain spv.* ops}}
    "dialect.op"() : () -> ()
  }
}

// -----

// Use external function
spv.module "Logical" "VulkanKHR" {
  // expected-error @+1 {{'spv.module' cannot contain external functions}}
  func @extern() -> ()
}

// -----

// Module with nested function
spv.module "Logical" "VulkanKHR" {
  func @outer_func() -> () {
    // expected-error @+1 {{'spv.module' cannot contain nested functions}}
    func @inner_func() -> () {
      spv.Return
    }
    spv.Return
  }
}

// -----

// expected-error @+1 {{uses unknown capability: MyAwesomeCapability}}
spv.module "Logical" "GLSL450" {
} attributes {
  capabilities = ["MyAwesomeCapability"]
}

// -----

// expected-error @+1 {{uses unknown extension: MyAwesomeExtension}}
spv.module "Logical" "GLSL450" {
} attributes {
  extensions = ["MyAwesomeExtension"]
}

// -----

//===----------------------------------------------------------------------===//
// spv._module_end
//===----------------------------------------------------------------------===//

func @module_end_not_in_module() -> () {
  // expected-error @+1 {{op must appear in a 'spv.module' block}}
  spv._module_end
}

// -----

//===----------------------------------------------------------------------===//
// spv._reference_of
//===----------------------------------------------------------------------===//

spv.module "Logical" "GLSL450" {
  spv.specConstant @sc1 = false
  spv.specConstant @sc2 = 42 : i64
  spv.specConstant @sc3 = 1.5 : f32

  // CHECK-LABEL: @reference
  func @reference() -> i1 {
    // CHECK: spv._reference_of @sc1 : i1
    %0 = spv._reference_of @sc1 : i1
    spv.ReturnValue %0 : i1
  }

  // CHECK-LABEL: @initialize
  func @initialize() -> i64 {
    // CHECK: spv._reference_of @sc2 : i64
    %0 = spv._reference_of @sc2 : i64
    %1 = spv.Variable init(%0) : !spv.ptr<i64, Function>
    %2 = spv.Load "Function" %1 : i64
    spv.ReturnValue %2 : i64
  }

  // CHECK-LABEL: @compute
  func @compute() -> f32 {
    // CHECK: spv._reference_of @sc3 : f32
    %0 = spv._reference_of @sc3 : f32
    %1 = spv.constant 6.0 : f32
    %2 = spv.FAdd %0, %1 : f32
    spv.ReturnValue %2 : f32
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    // expected-error @+1 {{expected spv.specConstant symbol}}
    %0 = spv._reference_of @sc : i32
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  spv.specConstant @sc = 42 : i32
  func @foo() -> () {
    // expected-error @+1 {{result type mismatch with the referenced specialization constant's type}}
    %0 = spv._reference_of @sc : f32
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.specConstant
//===----------------------------------------------------------------------===//

spv.module "Logical" "GLSL450" {
  spv.specConstant @sc1 = false
  spv.specConstant @sc2 = 42 : i64
  spv.specConstant @sc3 = 1.5 : f32
}

// -----

spv.module "Logical" "GLSL450" {
  // expected-error @+1 {{default value bitwidth disallowed}}
  spv.specConstant @sc = 15 : i4
}

// -----

spv.module "Logical" "GLSL450" {
  // expected-error @+1 {{default value can only be a bool, integer, or float scalar}}
  spv.specConstant @sc = dense<[2, 3]> : vector<2xi32>
}

// -----

func @use_in_function() -> () {
  // expected-error @+1 {{op must appear in a 'spv.module' block}}
  spv.specConstant @sc = false
  return
}
