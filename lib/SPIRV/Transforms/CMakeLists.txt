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
  MLIRSupport)
