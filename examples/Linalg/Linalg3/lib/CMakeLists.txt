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

add_llvm_library(Linalg3DialectConstruction
  DialectConstruction.cpp
  )
