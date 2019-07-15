add_llvm_library(MLIRControlFlowToCFG
  ConvertControlFlowToCFG.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ControlFlowToCFG
)
add_dependencies(
  MLIRControlFlowToCFG

  MLIRLoopOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRControlFlowToCFG

  MLIRLoopOps
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
