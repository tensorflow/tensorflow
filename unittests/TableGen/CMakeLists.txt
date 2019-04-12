add_mlir_unittest(MLIRTableGenTests
  FormatTest.cpp
)
target_link_libraries(MLIRDialectTests
  PRIVATE LLVMMLIRTableGen)
