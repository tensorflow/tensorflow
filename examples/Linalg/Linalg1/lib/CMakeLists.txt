set(LLVM_OPTIONAL_SOURCES
  Analysis.cpp
  ConvertToLLVMDialect.cpp
  SliceOp.cpp
  ViewOp.cpp
  Common.cpp
  Dialect.cpp
  RangeOp.cpp
  Utils.cpp
  ViewType.cpp
  DialectConstruction.cpp
  )

add_llvm_library(Linalg1
  Analysis.cpp
  ConvertToLLVMDialect.cpp
  SliceOp.cpp
  ViewOp.cpp
  Common.cpp
  Dialect.cpp
  RangeOp.cpp
  Utils.cpp
  ViewType.cpp
  )

target_link_libraries(Linalg1
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

add_llvm_library(Linalg1DialectConstruction
  DialectConstruction.cpp
  )

target_link_libraries(Linalg1DialectConstruction
  PUBLIC Linalg1)
