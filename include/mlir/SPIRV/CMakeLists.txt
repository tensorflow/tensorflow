set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVOps.h.inc -gen-op-decls)
mlir_tablegen(SPIRVOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSPIRVOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPIRVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSPIRVEnumsIncGen)

add_subdirectory(Transforms)
