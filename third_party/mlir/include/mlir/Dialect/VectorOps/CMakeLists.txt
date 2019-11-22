set(LLVM_TARGET_DEFINITIONS VectorOps.td)
mlir_tablegen(VectorOps.h.inc -gen-op-decls)
mlir_tablegen(VectorOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVectorOpsIncGen)

set(LLVM_TARGET_DEFINITIONS VectorTransformPatterns.td)
mlir_tablegen(VectorTransformPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRVectorTransformPatternsIncGen)
