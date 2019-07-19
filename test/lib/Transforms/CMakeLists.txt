add_llvm_library(MLIRTestTransforms
  TestConstantFold.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestVectorizationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )
add_dependencies(MLIRTestTransforms MLIRStandardOpsIncGen)
target_link_libraries(MLIRTestTransforms
  MLIRAffineOps
  MLIRAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRVectorOps
  )
