set(LLVM_TARGET_DEFINITIONS GPUOps.td)
mlir_tablegen(GPUOps.h.inc -gen-op-decls)
mlir_tablegen(GPUOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRGPUOpsIncGen)
