add_mlir_unittest(MLIRSPIRVTests
  DeserializationTest.cpp
)
target_link_libraries(MLIRSPIRVTests
  PRIVATE
  MLIRSPIRV
  MLIRSPIRVSerialization)

