file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRPass
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass
  )
add_dependencies(MLIRPass MLIRAnalysis MLIRIR LLVMSupport)
target_link_libraries(MLIRPass MLIRAnalysis MLIRIR LLVMSupport)
