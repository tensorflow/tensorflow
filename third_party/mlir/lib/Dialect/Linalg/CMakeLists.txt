add_llvm_library(MLIRLinalg
  LinalgRegistration.cpp
  Analysis/DependenceAnalysis.cpp
  IR/LinalgOps.cpp
  IR/LinalgTypes.cpp
  Transforms/Fusion.cpp
  Transforms/LowerToLLVMDialect.cpp
  Transforms/LowerToLoops.cpp
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
  MLIRLinalgOpsIncGen
  MLIRLinalgLibraryOpsIncGen
  MLIRStandardOps
  MLIRStandardToLLVM
  )
