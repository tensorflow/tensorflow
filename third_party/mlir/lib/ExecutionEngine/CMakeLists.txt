llvm_map_components_to_libnames(outlibs "nativecodegen" "IPO")
add_llvm_library(MLIRExecutionEngine
  ExecutionEngine.cpp
  OptUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ExecutionEngine
  )
target_link_libraries(MLIRExecutionEngine

  MLIRLLVMIR
  MLIRTargetLLVMIR
  LLVMBitReader
  LLVMBitWriter
  LLVMExecutionEngine
  LLVMOrcJIT
  LLVMSupport
  LLVMTransformUtils

  ${outlibs})
