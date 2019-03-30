add_mlir_unittest(MLIRDialectTests
  BroadcastShapeTest.cpp
)
target_link_libraries(MLIRDialectTests
  PRIVATE
  MLIRIR
  MLIRDialect)
