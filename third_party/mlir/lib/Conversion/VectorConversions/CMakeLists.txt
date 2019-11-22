add_llvm_library(MLIRVectorConversions
  VectorToLLVM.cpp
  VectorToLoops.cpp
  VectorToVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorConversions
)
set(LIBS
  MLIRLLVMIR
  MLIRTransforms
  LLVMCore
  LLVMSupport
  )

add_dependencies(MLIRVectorConversions ${LIBS})
add_dependencies(MLIRVectorConversions MLIRVectorTransformPatternsIncGen)
target_link_libraries(MLIRVectorConversions ${LIBS})
