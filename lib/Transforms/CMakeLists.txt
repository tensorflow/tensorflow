add_llvm_library(MLIRTransforms
  Canonicalizer.cpp
  CMakeLists.txt
  CSE.cpp
  DialectConversion.cpp
  DmaGeneration.cpp
  LoopFusion.cpp
  LoopTiling.cpp
  LoopUnrollAndJam.cpp
  LoopUnroll.cpp
  LowerAffine.cpp
  LowerVectorTransfers.cpp
  MaterializeVectors.cpp
  MemRefDataFlowOpt.cpp
  PipelineDataTransfer.cpp
  SimplifyAffineStructures.cpp
  StripDebugInfo.cpp
  TestConstantFold.cpp
  Utils/GreedyPatternRewriteDriver.cpp
  Utils/LoopUtils.cpp
  Utils/Utils.cpp
  Vectorization
  Vectorize.cpp
  Vectorization/VectorizerTestPass.cpp
  ViewFunctionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )
add_dependencies(MLIRTransforms MLIRStandardOpsIncGen)
target_link_libraries(MLIRTransforms MLIRVectorOps)
