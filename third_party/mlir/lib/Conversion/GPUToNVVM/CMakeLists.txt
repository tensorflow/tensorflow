set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
mlir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToNVVMIncGen)

add_llvm_library(MLIRGPUtoNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp
  )

add_dependencies(MLIRGPUtoNVVMTransforms
  MLIRGPUToNVVMIncGen)

target_link_libraries(MLIRGPUtoNVVMTransforms
  LLVMSupport
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  )
