set(LLVM_TARGET_DEFINITIONS LinalgOps.td)
mlir_tablegen(LinalgOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgOpsIncGen)
