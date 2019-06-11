add_mlir_unittest(MLIRIRTests
  AttributeTest.cpp
  DialectTest.cpp
  OperationSupportTest.cpp
)
target_link_libraries(MLIRIRTests
  PRIVATE
  MLIRIR)
