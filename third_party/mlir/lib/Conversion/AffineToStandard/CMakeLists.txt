add_llvm_library(MLIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStandard
)
add_dependencies(
  MLIRAffineToStandard

  MLIRAffineOps
  MLIRStandardOps
  MLIRIR
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRAffineToStandard

  MLIRAffineOps
  MLIRStandardOps
  MLIRIR
  LLVMCore
  LLVMSupport
)
