set(LLVM_TARGET_DEFINITIONS FxpMathOps.td)
mlir_tablegen(FxpMathOps.h.inc -gen-op-decls)
mlir_tablegen(FxpMathOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRFxpMathOpsIncGen)
