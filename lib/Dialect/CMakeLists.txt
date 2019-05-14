add_subdirectory(FxpMathOps)
add_subdirectory(QuantOps)

add_llvm_library(MLIRDialect
  Traits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect
  )
target_link_libraries(MLIRDialect MLIRIR)
