// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.EntryPoint
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // CHECK: {{%.*}} = spv.EntryPoint "GLCompute" @do_nothing
   %2 = spv.EntryPoint "GLCompute" @do_nothing
}

spv.module "Logical" "VulkanKHR" {
   %2 = spv.Variable : !spv.ptr<f32, Input>
   %3 = spv.Variable : !spv.ptr<f32, Output>
   func @do_something(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () {
     %1 = spv.Load "Input" %arg0 : f32
     spv.Store "Output" %arg1, %1 : f32
     spv.Return
   }
   // CHECK: {{%.*}} = spv.EntryPoint "GLCompute" @do_something, {{%.*}}, {{%.*}} : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>
   %4 = spv.EntryPoint "GLCompute" @do_something, %2, %3 : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{custom op 'spv.EntryPoint' expected function attribute}}
   %4 = spv.EntryPoint "GLCompute" "do_nothing"
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{function 'do_something' not found in 'spv.module'}}
   %4 = spv.EntryPoint "GLCompute" @do_something
}

/// TODO(ravishankarm) : Add a test that verifies an error is thrown
/// when interface entries of EntryPointOp are not
/// spv.Variables. There is currently no other op that has a spv.ptr
/// return type

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     // expected-error @+1 {{'spv.EntryPoint' op failed to verify that op can only be used in a 'spv.module' block}}
     %2 = spv.EntryPoint "GLCompute" @do_something
   }
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   %5 = spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{duplicate of a previous EntryPointOp}}
   %6 = spv.EntryPoint "GLCompute" @do_nothing
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   %5 = spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.EntryPoint' invalid execution_model attribute specification: "ContractionOff"}}
   %6 = spv.EntryPoint "ContractionOff" @do_nothing
}

// -----

spv.module "Logical" "VulkanKHR" {
   %2 = spv.Variable : !spv.ptr<f32, Workgroup>
   func @do_nothing() -> () {
     spv.Return
   }
   // expected-error @+1 {{'spv.EntryPoint' op invalid storage class 'Workgroup'}}
   %6 = spv.EntryPoint "GLCompute" @do_nothing, %2 : !spv.ptr<f32, Workgroup>
}

// -----

//===----------------------------------------------------------------------===//
// spv.ExecutionMode
//===----------------------------------------------------------------------===//

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   %7 = spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{%.*}} "ContractionOff"
   spv.ExecutionMode %7 "ContractionOff"
}

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   %8 = spv.EntryPoint "GLCompute" @do_nothing
   // CHECK: spv.ExecutionMode {{%.*}} "LocalSizeHint", 3, 4, 5
   spv.ExecutionMode %8 "LocalSizeHint", 3, 4, 5
}

// -----

spv.module "Logical" "VulkanKHR" {
   // expected-note @+1{{prior use here}}
   %2 = spv.Variable : !spv.ptr<f32, Input>
   func @do_nothing() -> () {
     spv.Return
   }
   %8 = spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{use of value '%2' expects different type than prior uses: '!spv.entrypoint' vs '!spv.ptr<f32, Input>'}}
   spv.ExecutionMode %2 "LocalSizeHint", 3, 4, 5
}

// -----

spv.module "Logical" "VulkanKHR" {
   func @do_nothing() -> () {
     spv.Return
   }
   %8 = spv.EntryPoint "GLCompute" @do_nothing
   // expected-error @+1 {{custom op 'spv.ExecutionMode' invalid execution_mode attribute specification: "GLCompute"}}
   spv.ExecutionMode %8 "GLCompute", 3, 4, 5
}

// -----

//===----------------------------------------------------------------------===//
// spv.FMul
//===----------------------------------------------------------------------===//

func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : f32
  return %0 : f32
}

func @fmul_vector(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = spv.FMul %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

func @fmul_i32(%arg: i32) -> i32 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : i32
  return %0 : i32
}

// -----

func @fmul_bf16(%arg: bf16) -> bf16 {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : bf16
  return %0 : bf16
}

// -----

func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{must be scalar/vector of 16/32/64-bit float}}
  %0 = spv.FMul %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.LoadOp
//===----------------------------------------------------------------------===//

// CHECK_LABEL: @simple_load
func @simple_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 : f32
  %1 = spv.Load "Function" %0 : f32
  return
}

// CHECK_LABEL: @volatile_load
func @volatile_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 ["Volatile"] : f32
  %1 = spv.Load "Function" %0 ["Volatile"] : f32
  return
}

// CHECK_LABEL: @aligned_load
func @aligned_load() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Load "Function" %0 ["Aligned", 4] : f32
  %1 = spv.Load "Function" %0 ["Aligned", 4] : f32
  return
}

// -----

func @simple_load_missing_storageclass() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected non-function type}}
  %1 = spv.Load %0 : f32
  return
}

// -----

func @simple_load_missing_operand() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected SSA operand}}
  %1 = spv.Load "Function" : f32
  return
}

// -----

func @simple_load_missing_rettype() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+2 {{expected ':'}}
  %1 = spv.Load "Function" %0
  return
}

// -----

func @volatile_load_missing_lbrace() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  %1 = spv.Load "Function" %0 "Volatile"] : f32
  return
}

// -----

func @volatile_load_missing_rbrace() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Volatile"} : f32
  return
}

// -----

func @aligned_load_missing_alignment() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spv.Load "Function" %0 ["Aligned"] : f32
  return
}

// -----

func @aligned_load_missing_comma() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  %1 = spv.Load "Function" %0 ["Aligned" 4] : f32
  return
}

// -----

func @load_incorrect_attributes() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Volatile", 4] : f32
  return
}

// -----

func @load_unknown_memory_access() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Load' invalid memory_access attribute specification: "Something"}}
  %1 = spv.Load "Function" %0 ["Something"] : f32
  return
}

// -----

func @aligned_load_incorrect_attributes() -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  %1 = spv.Load "Function" %0 ["Aligned", 4, 23] : f32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

"foo.function"() ({
  // expected-error @+1 {{must appear in a 'func' op}}
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
// spv.StoreOp
//===----------------------------------------------------------------------===//

func @simple_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 : f32
  spv.Store  "Function" %0, %arg0 : f32
  return
}

// CHECK_LABEL: @volatile_store
func @volatile_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  spv.Store  "Function" %0, %arg0 ["Volatile"] : f32
  return
}

// CHECK_LABEL: @aligned_store
func @aligned_store(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: spv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  spv.Store  "Function" %0, %arg0 ["Aligned", 4] : f32
  return
}

// -----

func @simple_store_missing_ptr_type(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected non-function type}}
  spv.Store  %0, %arg0 : f32
  return
}

// -----

func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Store' invalid operand}} : f32
  spv.Store  "Function" , %arg0 : f32
  return
}

// -----

func @simple_store_missing_operand(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{custom op 'spv.Store' expected 2 operands}} : f32
  spv.Store  "Function" %0 : f32
  return
}

// -----

func @volatile_store_missing_lbrace(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ':'}}
  spv.Store  "Function" %0, %arg0 "Volatile"] : f32
  return
}

// -----

func @volatile_store_missing_rbrace(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store "Function" %0, %arg0 ["Volatile"} : f32
  return
}

// -----

func @aligned_store_missing_alignment(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spv.Store  "Function" %0, %arg0 ["Aligned"] : f32
  return
}

// -----

func @aligned_store_missing_comma(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ','}}
  spv.Store  "Function" %0, %arg0 ["Aligned" 4] : f32
  return
}

// -----

func @load_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store  "Function" %0, %arg0 ["Volatile", 4] : f32
  return
}

// -----

func @aligned_store_incorrect_attributes(%arg0 : f32) -> () {
  %0 = spv.Variable : !spv.ptr<f32, Function>
  // expected-error @+1 {{expected ']'}}
  spv.Store  "Function" %0, %arg0 ["Aligned", 4, 23] : f32
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Variable
//===----------------------------------------------------------------------===//

func @variable_no_init(%arg0: f32) -> () {
  // CHECK: spv.Variable : !spv.ptr<f32, Function>
  %0 = spv.Variable : !spv.ptr<f32, Function>
  return
}

func @variable_init() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) : !spv.ptr<f32, Private>
  return
}

func @variable_bind() -> () {
  // CHECK: spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  %0 = spv.Variable bind(1, 2) : !spv.ptr<f32, Uniform>
  return
}

func @variable_init_bind() -> () {
  %0 = spv.constant 4.0 : f32
  // CHECK: spv.Variable init(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
  %1 = spv.Variable init(%0) {binding = 5 : i32} : !spv.ptr<f32, Private>
  return
}

// -----

func @expect_ptr_result_type(%arg0: f32) -> () {
  // expected-error @+1 {{expected spv.ptr type}}
  %0 = spv.Variable : f32
  return
}

// -----

func @variable_init(%arg0: f32) -> () {
  // expected-error @+1 {{op initializer must be the result of a spv.Constant or module-level spv.Variable op}}
  %0 = spv.Variable init(%arg0) : !spv.ptr<f32, Private>
  return
}

// -----

func @storage_class_mismatch() -> () {
  %0 = spv.constant 5.0 : f32
  // expected-error @+1 {{storage class must match result pointer's storage class}}
  %1 = "spv.Variable"(%0) {storage_class = 2: i32} : (f32) -> !spv.ptr<f32, Function>
  return
}

// -----

func @cannot_be_generic_storage_class(%arg0: f32) -> () {
  // expected-error @+1 {{storage class cannot be 'Generic'}}
  %0 = spv.Variable : !spv.ptr<f32, Generic>
  return
}
