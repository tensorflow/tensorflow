add_llvm_library(MLIRTestIR
  TestFunc.cpp
  TestSymbolUses.cpp

  ADDITIONAL_HEADER_DIRS
  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../TestDialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../TestDialect)
target_link_libraries(MLIRTestIR
  MLIRPass
  )
