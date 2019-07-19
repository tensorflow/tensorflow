add_llvm_library(MLIRAnalysis STATIC
  AffineAnalysis.cpp
  AffineStructures.cpp
  Dominance.cpp
  LoopAnalysis.cpp
  MemRefBoundCheck.cpp
  NestedMatcher.cpp
  OpStats.cpp
  SliceAnalysis.cpp
  TestMemRefDependenceCheck.cpp
  TestParallelismDetection.cpp
  Utils.cpp
  VectorAnalysis.cpp
  Verifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Analysis
  )
add_dependencies(MLIRAnalysis MLIRAffineOps MLIRLoopOps)
target_link_libraries(MLIRAnalysis MLIRAffineOps MLIRLoopOps)
