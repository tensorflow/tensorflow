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
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRLoopsToGPU
  MLIRLinalgToLLVM

  MLIRLoopToStandard
  MLIREDSC
  MLIRFxpMathOps
  MLIRGPU
  MLIRGPUtoNVVMTransforms
  MLIRGPUtoROCDLTransforms
  MLIRGPUtoSPIRVTransforms
  MLIRLinalg
  MLIRLLVMIR
  MLIRLoopOps
  MLIRNVVMIR
  MLIROptMain
  MLIRParser
  MLIRPass
  MLIRQuantizerTransforms
  MLIRQuantOps
  MLIRROCDLIR
  MLIRSPIRV
  MLIRStandardToSPIRVTransforms
  MLIRSPIRVTransforms
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRTestDialect
  MLIRTestIR
  MLIRTestPass
  MLIRTestTransforms
  MLIRSupport
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
)
if(MLIR_CUDA_CONVERSIONS_ENABLED)
  list(APPEND LIBS
    MLIRGPUtoCUDATransforms
  )
endif()
add_llvm_tool(mlir-opt
 mlir-opt.cpp
)
llvm_update_compile_flags(mlir-opt)
whole_archive_link(mlir-opt ${LIBS})
target_link_libraries(mlir-opt PRIVATE MLIRIR MLIRMlirOptLib ${LIBS} LLVMSupport)
