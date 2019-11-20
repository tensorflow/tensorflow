add_llvm_library(MLIRTestIR
  TestFunc.cpp
  TestSymbolUses.cpp

  ADDITIONAL_HEADER_DIRS
  )
target_link_libraries(MLIRTestIR
  MLIRPass
  )
