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
  DialectRegistration.cpp
  )

set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIRControlFlowToCFG
  MLIREDSC
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTransforms
)

add_llvm_library(Linalg1LLVMConversion
  ConvertToLLVMDialect.cpp
  )
target_link_libraries(Linalg1LLVMConversion PUBLIC MLIRLLVMIR
  MLIRControlFlowToCFG MLIRStandardOps)

add_llvm_library(Linalg1
  Analysis.cpp
  SliceOp.cpp
  ViewOp.cpp
  Common.cpp
  Dialect.cpp
  RangeOp.cpp
  Utils.cpp
  ViewType.cpp
  DEPENDS
  intrinsics_gen
  )
target_link_libraries(Linalg1
  PUBLIC
    ${LIBS}
    Linalg1LLVMConversion
  )

add_llvm_library(Linalg1DialectConstruction
  DialectConstruction.cpp
  )
target_link_libraries(Linalg1DialectConstruction PUBLIC Linalg1)

add_llvm_executable(linalg1-opt
  DialectRegistration.cpp
  )
llvm_update_compile_flags(linalg1-opt)
whole_archive_link(linalg1-opt
  Linalg1LLVMConversion
  Linalg1DialectConstruction
  ${LIBS}
  )
target_link_libraries(linalg1-opt
  PRIVATE
  Linalg1
  Linalg1LLVMConversion
  Linalg1DialectConstruction
  MLIRLLVMIR
  MLIRMlirOptLib
  MLIROptMain
  ${LIBS}
  LLVMSupport)
