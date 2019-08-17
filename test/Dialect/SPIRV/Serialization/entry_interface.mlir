// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_loadstore() -> () {
  spv.module "Logical" "VulkanKHR" {
    // CHECK:       spv.globalVariable !spv.ptr<f32, Input> @var2
    // CHECK-NEXT:  spv.globalVariable !spv.ptr<f32, Output> @var3
    // CHECK-NEXT:  func @noop({{%.*}}: !spv.ptr<f32, Input>, {{%.*}}: !spv.ptr<f32, Output>)
    // CHECK:       spv.EntryPoint "GLCompute" @noop, @var2, @var3
    spv.globalVariable !spv.ptr<f32, Input> @var2
    spv.globalVariable !spv.ptr<f32, Output> @var3
    func @noop(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () {
      spv.Return
    }
    spv.EntryPoint "GLCompute" @noop, @var2, @var3
    spv.ExecutionMode @noop "ContractionOff"
  }
  return
}