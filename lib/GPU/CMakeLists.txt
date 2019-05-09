add_llvm_library(MLIRGPU
  IR/GPUDialect.cpp
  IR/DialectRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/GPU
)
add_dependencies(MLIRGPU MLIRGPUOpsIncGen MLIRIR LLVMSupport)
target_link_libraries(MLIRGPU MLIRIR LLVMSupport)
