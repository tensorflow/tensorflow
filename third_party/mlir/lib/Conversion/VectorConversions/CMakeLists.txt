add_llvm_library(MLIRVectorToLLVM
  VectorToLLVM.cpp
  VectorToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorConversions
)
set(LIBS
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )

add_dependencies(MLIRVectorToLLVM ${LIBS})
target_link_libraries(MLIRVectorToLLVM ${LIBS})
