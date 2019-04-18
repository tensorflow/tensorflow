add_mlir_unittest(MLIRTableGenTests
  FormatTest.cpp
)
target_link_libraries(MLIRTableGenTests
  PRIVATE LLVMMLIRTableGen)
