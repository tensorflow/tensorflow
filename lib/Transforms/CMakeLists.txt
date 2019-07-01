add_subdirectory(Utils)

add_llvm_library(MLIRTransforms
  Canonicalizer.cpp
  CSE.cpp
  DialectConversion.cpp
  DmaGeneration.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
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
  Vectorize.cpp
  ViewRegionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )

add_dependencies(MLIRTransforms MLIRStandardOpsIncGen)
target_link_libraries(MLIRTransforms
  MLIRAffineOps
  MLIRAnalysis
  MLIRPass
  MLIRTransformUtils
  MLIRVectorOps
  )
