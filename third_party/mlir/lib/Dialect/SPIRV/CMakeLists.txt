add_llvm_library(MLIRSPIRV
  DialectRegistration.cpp
  LayoutUtils.cpp
  SPIRVDialect.cpp
  SPIRVOps.cpp
  SPIRVLowering.cpp
  SPIRVTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SPIRV
  )

add_dependencies(MLIRSPIRV
  MLIRSPIRVOpsIncGen
  MLIRSPIRVEnumsIncGen
  MLIRSPIRVOpUtilsGen)

target_link_libraries(MLIRSPIRV
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms)

add_subdirectory(Serialization)
add_subdirectory(Transforms)
