// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {gpu.container_module} {

  module @kernels attributes {gpu.kernel_module} {
    // CHECK:       spv.module "Logical" "GLSL450" {
    // CHECK-DAG:    spv.globalVariable [[VAR0:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<f32 [0]>, StorageBuffer>
    // CHECK-DAG:    spv.globalVariable [[VAR1:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x f32 [4]> [0]>, StorageBuffer>
    // CHECK:    func [[FN:@.*]]()
    func @kernel_1(%arg0 : f32, %arg1 : memref<12xf32, 1>)
        attributes { gpu.kernel } {
      // CHECK: [[ADDRESSARG0:%.*]] = spv._address_of [[VAR0]]
      // CHECK: [[CONST0:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG0PTR:%.*]] = spv.AccessChain [[ADDRESSARG0]]{{\[}}[[CONST0]]
      // CHECK: [[ARG0:%.*]] = spv.Load "StorageBuffer" [[ARG0PTR]]
      // CHECK: [[ADDRESSARG1:%.*]] = spv._address_of [[VAR1]]
      // CHECK: [[CONST1:%.*]] = spv.constant 0 : i32
      // CHECK: [[ARG1:%.*]] = spv.AccessChain [[ADDRESSARG1]]{{\[}}[[CONST1]]
      // CHECK-NEXT: spv.Return
      // CHECK: spv.EntryPoint "GLCompute" [[FN]]
      // CHECK: spv.ExecutionMode [[FN]] "LocalSize"
      return
    }
  }

  func @foo() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32, 1>)
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = "kernel_1", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<12xf32, 1>) -> ()
    return
  }
}
