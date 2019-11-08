add_llvm_library(MLIRGPU
  IR/GPUDialect.cpp
  IR/DialectRegistration.cpp
  Transforms/KernelOutlining.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU
)
add_dependencies(MLIRGPU MLIRGPUOpsIncGen MLIRIR MLIRLLVMIR LLVMSupport)
target_link_libraries(MLIRGPU MLIRIR MLIRLLVMIR MLIRStandardOps LLVMSupport)
