// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_loadstore() -> () {
  spv.module "Logical" "VulkanKHR" {
    // CHECK:       [[VAR1:%.*]] = spv.Variable : !spv.ptr<f32, Input>
    // CHECK-NEXT:  [[VAR2:%.*]] = spv.Variable : !spv.ptr<f32, Output>
    // CHECK-NEXT:  func @noop({{%.*}}: !spv.ptr<f32, Input>, {{%.*}}: !spv.ptr<f32, Output>)
    // CHECK:       spv.EntryPoint "GLCompute" @noop, [[VAR1]], [[VAR2]]
    %2 = spv.Variable : !spv.ptr<f32, Input>
    %3 = spv.Variable : !spv.ptr<f32, Output>
    func @noop(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) -> () {
      spv.Return
    }
    spv.EntryPoint "GLCompute" @noop, %2, %3 : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>
    spv.ExecutionMode @noop "ContractionOff"
  }
  return
}