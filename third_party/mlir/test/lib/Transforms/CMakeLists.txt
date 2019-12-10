add_llvm_library(MLIRTestTransforms
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestLoopFusion.cpp
  TestInlining.cpp
  TestLinalgTransforms.cpp
  TestLinalgTilePermuteTransforms.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestOpaqueLoc.cpp
  TestMemRefStrideCalculation.cpp
  TestVectorToLoopsConversion.cpp
  TestVectorTransforms.cpp
  TestVectorizationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../TestDialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../TestDialect)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../DeclarativeTransforms)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../DeclarativeTransforms)
add_dependencies(MLIRTestTransforms MLIRStandardOpsIncGen)
add_dependencies(MLIRTestTransforms MLIRTestLinalgTransformPatternsIncGen)
add_dependencies(MLIRTestTransforms MLIRTestLinalgTilePermutePatternsIncGen)
add_dependencies(MLIRTestTransforms MLIRTestVectorTransformPatternsIncGen)
target_link_libraries(MLIRTestTransforms
  MLIRAffineOps
  MLIRAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRTestDialect
  MLIRVectorOps
  )
