set(LLVM_TARGET_DEFINITIONS QuantOps.td)
mlir_tablegen(QuantOps.h.inc -gen-op-decls)
mlir_tablegen(QuantOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRQuantOpsIncGen)
