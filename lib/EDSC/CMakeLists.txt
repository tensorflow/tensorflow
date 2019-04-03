add_llvm_library(MLIREDSC
  Builders.cpp
  CoreAPIs.cpp
  Helpers.cpp
  Intrinsics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC
  )
add_dependencies(MLIREDSC MLIRReferenceImplementationTestGen)
