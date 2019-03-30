set(LLVM_TARGET_DEFINITIONS "${MLIR_MAIN_SRC_DIR}/../test/mlir-tblgen/reference-impl.td")
mlir_tablegen("reference-impl.inc" -gen-reference-implementations)
add_public_tablegen_target(MLIRReferenceImplementationTestGen)
