add_llvm_library(MLIRFxpMathOps
  IR/FxpMathOps.cpp
  IR/DialectRegistration.cpp
  Transforms/LowerUniformRealMath.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/FxpMathOps
  )
add_dependencies(MLIRFxpMathOps
                 MLIRFxpMathOpsIncGen
                 MLIRQuantOps
                 MLIRIR
                 MLIRPass
                 MLIRSupport
                 MLIRStandardOps)
