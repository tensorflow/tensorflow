add_mlir_dialect(LinalgOps LinalgDoc)
set(LLVM_TARGET_DEFINITIONS LinalgStructuredOps.td)
mlir_tablegen(LinalgStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgStructuredOps.cpp.inc -gen-op-defs)
mlir_tablegen(LinalgStructuredOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LinalgStructuredOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLinalgStructuredOpsIncGen)

