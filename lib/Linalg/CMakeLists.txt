add_llvm_library(MLIRLinalg
  LinalgRegistration.cpp
  IR/LinalgOps.cpp
  IR/LinalgTypes.cpp
  Transforms/LowerToLLVMDialect.cpp
  Transforms/Tiling.cpp
  Utils/Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Linalg
  )
add_dependencies(MLIRLinalg MLIRLinalgOpsIncGen)
