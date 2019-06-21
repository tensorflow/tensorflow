set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTargetNVVMIR
  MLIRTransforms
  MLIRTranslation
  MLIRSupport
  MLIRVectorOps
)
add_llvm_executable(mlir-translate
  mlir-translate.cpp
)
llvm_update_compile_flags(mlir-translate)
whole_archive_link(mlir-translate ${LIBS})
target_link_libraries(mlir-translate PRIVATE MLIRIR ${LIBS} LLVMSupport)
