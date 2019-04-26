add_llvm_library(MLIRSupport
  FileUtilities.cpp
  StorageUniquer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
  )
target_link_libraries(MLIRSupport LLVMSupport)
