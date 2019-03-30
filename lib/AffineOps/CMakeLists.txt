add_llvm_library(MLIRAffineOps
  AffineOps.cpp
  DialectRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/AffineOps
  )
add_dependencies(MLIRAffineOps MLIRIR MLIRStandardOps)
target_link_libraries(MLIRAffineOps MLIRIR MLIRStandardOps)

