set(LLVM_OPTIONAL_SOURCES
  mlir-cpu-runner-lib.cpp
  mlir-cpu-runner.cpp
  )

set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIRControlFlowToCFG
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRLLVMIR
  MLIRParser
  MLIRTargetLLVMIR
  MLIRTransforms
  MLIRStandardToLLVM
  MLIRSupport
  LLVMCore
  LLVMSupport
)

add_llvm_library(MLIRCPURunnerLib
  mlir-cpu-runner-lib.cpp
)
target_link_libraries(MLIRCPURunnerLib ${LIBS})

add_llvm_executable(mlir-cpu-runner
  mlir-cpu-runner.cpp
)
llvm_update_compile_flags(mlir-cpu-runner)
whole_archive_link(mlir-cpu-runner MLIRLLVMIR MLIRStandardOps MLIRTargetLLVMIR MLIRTransforms MLIRTranslation)
target_link_libraries(mlir-cpu-runner PRIVATE MLIRIR ${LIBS} MLIRCPURunnerLib)
