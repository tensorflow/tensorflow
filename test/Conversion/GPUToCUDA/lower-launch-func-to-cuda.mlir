// RUN: mlir-opt %s --launch-func-to-cuda | FileCheck %s

// CHECK: llvm.global constant @[[kernel_name:.*]]("kernel\00")

func @cubin_getter() -> !llvm<"i8*">

func @kernel(!llvm.float, !llvm<"float*">)
    attributes { gpu.kernel, nvvm.cubingetter = @cubin_getter }


func @foo() {
  %0 = "op"() : () -> (!llvm.float)
  %1 = "op"() : () -> (!llvm<"float*">)
  %cst = constant 8 : index

  // CHECK: [[module_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mcuModuleLoad([[module_ptr]], {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: [[func_ptr:%.*]] = llvm.alloca {{.*}} x !llvm<"i8*"> : (!llvm.i32) -> !llvm<"i8**">
  // CHECK: llvm.call @mcuModuleGetFunction([[func_ptr]], {{.*}}, {{.*}}) : (!llvm<"i8**">, !llvm<"i8*">, !llvm<"i8*">) -> !llvm.i32
  // CHECK: llvm.call @mcuGetStreamHelper
  // CHECK: llvm.call @mcuLaunchKernel
  // CHECK: llvm.call @mcuStreamSynchronize
  "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = @kernel }
      : (index, index, index, index, index, index, !llvm.float, !llvm<"float*">) -> ()

  return
}
