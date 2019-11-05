add_llvm_library(MLIRSPIRV
  DialectRegistration.cpp
  SPIRVDialect.cpp
  SPIRVOps.cpp
  SPIRVTypes.cpp
  LayoutUtils.cpp

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
  MLIRSupport)

add_subdirectory(Serialization)
add_subdirectory(Transforms)
