set(LLVM_TARGET_DEFINITIONS StdOpsToSPIRVConversion.td)
mlir_tablegen(StdOpsToSPIRVConversion.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStdOpsToSPIRVConversionIncGen)

add_llvm_library(MLIRSPIRVConversion
  StdOpsToSPIRVConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SPIRV
  )

add_dependencies(MLIRSPIRVConversion
  MLIRStdOpsToSPIRVConversionIncGen)

target_link_libraries(MLIRSPIRVConversion
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransformUtils)
