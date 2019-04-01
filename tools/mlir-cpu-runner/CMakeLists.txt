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
llvm_update_compile_flags(mlir-cpu-runner)
whole_archive_link(mlir-cpu-runner MLIRLLVMIR MLIRStandardOps MLIRTargetLLVMIR MLIRTransforms MLIRTranslation)
target_link_libraries(mlir-cpu-runner MLIRIR ${LIBS})
