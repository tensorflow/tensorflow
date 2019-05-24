add_mlir_unittest(MLIRIRTests
  DialectTest.cpp
  OperationSupportTest.cpp
)
target_link_libraries(MLIRIRTests
  PRIVATE
  MLIRIR)
