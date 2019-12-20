add_llvm_library(MLIRLinalg
  LinalgRegistration.cpp
  Analysis/DependenceAnalysis.cpp
  EDSC/Builders.cpp
  IR/LinalgOps.cpp
  IR/LinalgTypes.cpp
  Transforms/Fusion.cpp
  Transforms/LinalgTransforms.cpp
  Transforms/LinalgToLoops.cpp
  Transforms/Promotion.cpp
  Transforms/Tiling.cpp
  Utils/Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  DEPENDS
  intrinsics_gen
  )

add_dependencies(MLIRLinalg

  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRLinalgOpsIncGen
  MLIRLinalgStructuredOpsIncGen
  MLIRLinalgTransformPatternsIncGen
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRVectorOps
  )
