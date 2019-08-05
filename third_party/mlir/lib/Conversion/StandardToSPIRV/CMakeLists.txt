set(LLVM_TARGET_DEFINITIONS StandardToSPIRV.td)
mlir_tablegen(StandardToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStandardToSPIRVIncGen)

add_llvm_library(MLIRSPIRVConversion
  ConvertStandardToSPIRV.cpp
  ConvertStandardToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  )

add_dependencies(MLIRSPIRVConversion
  MLIRStandardToSPIRVIncGen)

target_link_libraries(MLIRSPIRVConversion
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransformUtils
  MLIRSPIRV
  MLIRStandardOps
  )
