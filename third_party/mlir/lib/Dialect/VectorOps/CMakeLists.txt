add_llvm_library(MLIRVectorOps
  DialectRegistration.cpp
  VectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/VectorOps
  )

add_dependencies(MLIRVectorOps MLIRVectorOpsIncGen)

target_link_libraries(MLIRVectorOps MLIRIR)
