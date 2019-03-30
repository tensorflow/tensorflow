set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRParser
  MLIRPass
  MLIRStandardOps
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRTranslation
  MLIRSupport
  MLIRVectorOps
)
add_executable(mlir-translate
  mlir-translate.cpp
)

whole_archive_link(mlir-translate ${LIBS})
target_link_libraries(mlir-translate MLIRIR ${LIBS} LLVMSupport)
