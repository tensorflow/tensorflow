set(LLVM_TARGET_DEFINITIONS SPIRVLowering.td)
mlir_tablegen(SPIRVLowering.h.inc -gen-struct-attr-decls)
mlir_tablegen(SPIRVLowering.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRSPIRVLoweringStructGen)

add_mlir_dialect(SPIRVOps SPIRVOps)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVEnums.h.inc -gen-enum-decls)
mlir_tablegen(SPIRVEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRSPIRVEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS SPIRVOps.td)
mlir_tablegen(SPIRVSerialization.inc -gen-spirv-serialization)
add_public_tablegen_target(MLIRSPIRVSerializationGen)

set(LLVM_TARGET_DEFINITIONS SPIRVBase.td)
mlir_tablegen(SPIRVOpUtils.inc -gen-spirv-op-utils)
add_public_tablegen_target(MLIRSPIRVOpUtilsGen)
