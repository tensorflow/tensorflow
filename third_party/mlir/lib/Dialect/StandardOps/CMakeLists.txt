file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRStandardOps
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/StandardOps
  )
add_dependencies(MLIRStandardOps MLIRStandardOpsIncGen LLVMSupport)
target_link_libraries(MLIRStandardOps LLVMSupport)
