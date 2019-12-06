add_llvm_library(MLIRTestIR
  TestSymbolUses.cpp

  ADDITIONAL_HEADER_DIRS
  )
target_link_libraries(MLIRTestIR
  MLIRPass
  )
