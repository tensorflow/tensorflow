set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

set(LIB_LIBS
  MLIRAnalysis
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
add_llvm_library(MLIRMlirOptLib
  mlir-opt.cpp
)
target_link_libraries(MLIRMlirOptLib ${LIB_LIBS})

set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRLinalg
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRQuantization
  MLIRStandardOps
  MLIRTransforms
  MLIRSupport
  MLIRVectorOps
)
add_executable(mlir-opt
 mlir-opt.cpp
)
llvm_update_compile_flags(mlir-opt)
whole_archive_link(mlir-opt ${LIBS})
target_link_libraries(mlir-opt MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)
