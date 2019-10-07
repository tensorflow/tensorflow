add_llvm_library(MLIRTestTransforms
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestLoopFusion.cpp
  TestInlining.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestOpaqueLoc.cpp
  TestMemRefStrideCalculation.cpp
  TestVectorizationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../TestDialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../TestDialect)
add_dependencies(MLIRTestTransforms MLIRStandardOpsIncGen)
target_link_libraries(MLIRTestTransforms
  MLIRAffineOps
  MLIRAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRTestDialect
  MLIRVectorOps
  )
