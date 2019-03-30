set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRTransforms
  MLIRSupport
  LLVMCore
  LLVMSupport
)
add_executable(mlir-cpu-runner
  mlir-cpu-runner.cpp
)

whole_archive_link(mlir-cpu-runner MLIRLLVMIR MLIRStandardOps MLIRTargetLLVMIR MLIRTransforms MLIRTranslation)
target_link_libraries(mlir-cpu-runner MLIRIR ${LIBS})
