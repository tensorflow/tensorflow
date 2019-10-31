// RUN: mlir-opt %s -lower-gpu-ops-to-rocdl-ops -split-input-file | FileCheck %s

module attributes {gpu.kernel_module} {
  // CHECK-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      attributes { gpu.kernel } {
    // CHECK: rocdl.workitem.id.x : !llvm.i32
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workitem.id.y : !llvm.i32
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workitem.id.z : !llvm.i32
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.dim.x : !llvm.i32
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.y : !llvm.i32
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.z : !llvm.i32
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.id.x : !llvm.i32
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.id.y : !llvm.i32
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.id.z : !llvm.i32
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.grid.dim.x : !llvm.i32
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.grid.dim.y : !llvm.i32
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.grid.dim.z : !llvm.i32
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

    std.return
  }
}

// -----

module attributes {gpu.kernel_module} {
  // CHECK: llvm.func @_ocml_exp_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @_ocml_exp_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) {
    %exp_f32 = std.exp %arg_f32 : f32
    // CHECK: llvm.call @_ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result_f32 = std.exp %exp_f32 : f32
    // CHECK: llvm.call @_ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.exp %arg_f64 : f64
    // CHECK: llvm.call @_ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return
  }
}


// -----

// Test that we handled properly operation with SymbolTable other than module op
module attributes {gpu.kernel_module} {
  "test.symbol_scope"() ({
    // CHECK: test.symbol_scope
    // CHECK: llvm.func @_ocml_exp_f32(!llvm.float) -> !llvm.float
    // CHECK: llvm.func @_ocml_exp_f64(!llvm.double) -> !llvm.double
    // CHECK-LABEL: func @gpu_exp
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) {
      %exp_f32 = std.exp %arg_f32 : f32
      // CHECK: llvm.call @_ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result_f32 = std.exp %exp_f32 : f32
      // CHECK: llvm.call @_ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result64 = std.exp %arg_f64 : f64
      // CHECK: llvm.call @_ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
      std.return
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}
