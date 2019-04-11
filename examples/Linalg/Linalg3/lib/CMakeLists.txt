set(LLVM_OPTIONAL_SOURCES
  Analysis.cpp
  ConvertToLLVMDialect.cpp
  LoadStoreOps.cpp
  Transforms.cpp
  DialectConstruction.cpp
  TensorOps.cpp
  )

add_llvm_library(Linalg3
  Analysis.cpp
  ConvertToLLVMDialect.cpp
  LoadStoreOps.cpp
  Transforms.cpp
  TensorOps.cpp
  )

target_link_libraries(Linalg3
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

add_llvm_library(Linalg3DialectConstruction
  DialectConstruction.cpp
  )

target_link_libraries(Linalg3DialectConstruction
  PUBLIC Linalg3)
