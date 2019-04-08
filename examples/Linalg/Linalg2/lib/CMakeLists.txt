set(LLVM_OPTIONAL_SOURCES
  DialectConstruction.cpp
  TensorOps.cpp
  Transforms.cpp
  )

add_llvm_library(Linalg2
  TensorOps.cpp
  Transforms.cpp
  )

add_llvm_library(Linalg2DialectConstruction
  DialectConstruction.cpp
  )
