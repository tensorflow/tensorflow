set(LLVM_OPTIONAL_SOURCES
  DialectConstruction.cpp
  TensorOps.cpp
  Transforms.cpp
  )

add_llvm_library(Linalg2
  TensorOps.cpp
  Transforms.cpp
  )

target_link_libraries(Linalg2
  PUBLIC
    MLIRAnalysis
    MLIRDialect
    MLIREDSC
    MLIRIR
    MLIRLLVMIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    )

add_llvm_library(Linalg2DialectConstruction
  DialectConstruction.cpp
  )

target_link_libraries(Linalg2DialectConstruction
  PUBLIC Linalg2)
