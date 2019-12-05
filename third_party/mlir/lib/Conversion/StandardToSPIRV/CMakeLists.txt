set(LLVM_TARGET_DEFINITIONS StandardToSPIRV.td)
mlir_tablegen(StandardToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRStandardToSPIRVIncGen)

add_llvm_library(MLIRStandardToSPIRVTransforms
  ConvertStandardToSPIRV.cpp
  ConvertStandardToSPIRVPass.cpp
  LegalizeStandardForSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  )

add_dependencies(MLIRStandardToSPIRVTransforms
  MLIRStandardToSPIRVIncGen)

target_link_libraries(MLIRStandardToSPIRVTransforms
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  MLIRTransformUtils
  MLIRSPIRV
  MLIRStandardOps
  )
