// RUN: mlir-hlo-opt %s -gpu-kernel-to-nvvm | FileCheck %s

gpu.module @test_module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  gpu.func @test_kernel() kernel {
    %0 = gpu.block_id x
    gpu.return
  }
}

// CHECK-LABEL:  gpu.module @test_module
// CHECK-SAME:     attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
// CHECK-NEXT:    llvm.func @test_kernel() attributes {gpu.kernel, nvvm.kernel} {
// CHECK-NEXT:      %0 = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK-NEXT:      llvm.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
