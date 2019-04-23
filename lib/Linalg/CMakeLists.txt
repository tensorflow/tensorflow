add_llvm_library(MLIRLinalg
  LinalgOps.cpp
  LinalgRegistration.cpp
  LinalgTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Linalg
  )
add_dependencies(MLIRLinalg MLIRLinalgOpsIncGen)
