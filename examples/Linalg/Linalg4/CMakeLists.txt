add_subdirectory(lib)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LLVM_OPTIONAL_SOURCES Example.cpp)

add_llvm_example(linalg-example-4
  Example.cpp
  )

target_link_libraries(linalg-example-4
  PRIVATE
    MLIRAnalysis
    MLIRDialect
    MLIREDSC
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    Linalg1
    Linalg2
    Linalg3
    Linalg4
    Linalg3DialectConstruction
    )

whole_archive_link(linalg-example-4
  MLIRStandardOps
  )

