set(LLVM_TARGET_DEFINITIONS LoopOps.td)
mlir_tablegen(LoopOps.h.inc -gen-op-decls)
mlir_tablegen(LoopOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLoopOpsIncGen)
