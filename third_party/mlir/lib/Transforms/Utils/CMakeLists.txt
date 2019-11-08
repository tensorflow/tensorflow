add_llvm_library(MLIRTransformUtils
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopFusionUtils.cpp
  LoopUtils.cpp
  RegionUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  )

add_dependencies(MLIRTransformUtils MLIRStandardOpsIncGen)
target_link_libraries(MLIRTransformUtils
  MLIRAffineOps
  MLIRAnalysis
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  )
