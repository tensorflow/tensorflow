// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:           func {{@.*}}([[ARG1:%.*]]: !spv.ptr<f32, Input>, [[ARG2:%.*]]: !spv.ptr<f32, Output>) {
// CHECK-NEXT:        [[VALUE:%.*]] = spv.Load "Input" [[ARG1]] : f32
// CHECK-NEXT:        spv.Store "Output" [[ARG2]], [[VALUE]] : f32

func @spirv_loadstore() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @load_store(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) {
      %1 = spv.Load "Input" %arg0 : f32
      spv.Store "Output" %arg1, %1 : f32
      spv.Return
    }
  }
  return
}