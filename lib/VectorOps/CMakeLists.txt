add_llvm_library(MLIRVectorOps
  DialectRegistration.cpp
  VectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/VectorOps
  )
