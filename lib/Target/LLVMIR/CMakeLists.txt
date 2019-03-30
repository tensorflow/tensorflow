add_llvm_library(MLIRTargetLLVMIR
  ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(MLIRTargetLLVMIR MLIRLLVMIR MLIRTranslation LLVMCore LLVMSupport LLVMTransformUtils)
