set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVOps.h.inc -gen-op-decls)
mlir_tablegen(SPIRVOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSPIRVOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVStructureOps.td)
mlir_tablegen(SPIRVStructureOps.h.inc -gen-op-decls)
mlir_tablegen(SPIRVStructureOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSPIRVStructureOpsIncGen)
