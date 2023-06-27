// RUN: mlir-hlo-opt %s -gpu-kernel-to-nvvm | FileCheck %s

gpu.module @test_module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  gpu.func @test_kernel(%out: memref<32xf32>) kernel {
    %0 = gpu.block_id x
    %cst = arith.constant 0.0 : f32
    memref.store %cst, %out[%0] : memref<32xf32>
    gpu.return
  }
}

// CHECK-LABEL:  gpu.module @test_module
// CHECK-SAME:     attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
// CHECK-NEXT:    llvm.func @test_kernel
// CHECK-SAME         attributes {gpu.kernel, nvvm.kernel}
// CHECK:           %[[VAR:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
