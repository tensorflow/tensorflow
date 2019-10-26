set(LLVM_OPTIONAL_SOURCES
  Builders.cpp
  CoreAPIs.cpp
  Helpers.cpp
  Intrinsics.cpp
  )

add_llvm_library(MLIREDSC
  Builders.cpp
  Helpers.cpp
  Intrinsics.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC
  )
add_dependencies(MLIREDSC MLIRReferenceImplementationTestGen)
target_link_libraries(MLIREDSC
  PUBLIC
    MLIRAffineOps
    MLIRLoopOps
    MLIRStandardOps
    MLIRTransformUtils
    MLIRVectorOps
    )

add_llvm_library(MLIREDSCInterface
  CoreAPIs.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/EDSC
  )
add_dependencies(MLIREDSCInterface MLIRIR)
add_dependencies(MLIREDSC MLIRReferenceImplementationTestGen)
target_link_libraries(MLIREDSC
  PUBLIC
    MLIRIR
    MLIRSupport
    MLIRParser
    )
