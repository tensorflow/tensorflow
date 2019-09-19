set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVOps.h.inc -gen-op-decls)
mlir_tablegen(SPIRVOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSPIRVOpsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVIntEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPIRVIntEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSPIRVIntEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVBitEnums.h.inc -gen-spirv-enum-decls)
mlir_tablegen(SPIRVBitEnums.cpp.inc -gen-spirv-enum-defs)
add_public_tablegen_target(MLIRSPIRVBitEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVSerialization.inc -gen-spirv-serialization)
add_public_tablegen_target(MLIRSPIRVSerializationGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVOpUtils.inc -gen-spirv-op-utils)
add_public_tablegen_target(MLIRSPIRVOpUtilsGen)
