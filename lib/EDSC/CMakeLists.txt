add_llvm_library(MLIREDSC
  Builders.cpp
  Helpers.cpp
  Intrinsics.cpp
  MLIREmitter.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC
  )
add_dependencies(MLIREDSC MLIRReferenceImplementationTestGen)
