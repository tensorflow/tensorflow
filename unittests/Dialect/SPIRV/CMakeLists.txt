add_mlir_unittest(MLIRSPIRVTests
  DeserializationTest.cpp
  SerializationTest.cpp
)
target_link_libraries(MLIRSPIRVTests
  PRIVATE
  MLIRSPIRV
  MLIRSPIRVSerialization)

whole_archive_link(MLIRSPIRVTests MLIRSPIRV)

