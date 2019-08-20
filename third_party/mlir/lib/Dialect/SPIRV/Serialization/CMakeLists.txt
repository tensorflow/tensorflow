add_llvm_library(MLIRSPIRVSerialization
  ConvertFromBinary.cpp
  ConvertToBinary.cpp
  Deserializer.cpp
  Serializer.cpp
  SPIRVBinaryUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SPIRV
  )

add_dependencies(MLIRSPIRVSerialization
  MLIRSPIRVSerializationGen)

target_link_libraries(MLIRSPIRVSerialization
  MLIRIR
  MLIRSPIRV
  MLIRSupport)
