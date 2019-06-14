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
  MLIRAffineToGPU
  MLIRAnalysis
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRLinalg
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRSPIRV
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRSupport
  MLIRVectorOps
)
add_llvm_executable(mlir-opt
 mlir-opt.cpp
)
llvm_update_compile_flags(mlir-opt)
whole_archive_link(mlir-opt ${LIBS})
target_link_libraries(mlir-opt PRIVATE MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)
