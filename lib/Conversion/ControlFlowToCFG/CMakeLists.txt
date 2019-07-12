add_llvm_library(MLIRControlFlowToCFG
  ConvertControlFlowToCFG.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ControlFlowToCFG
)
add_dependencies(
  MLIRControlFlowToCFG

  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRControlFlowToCFG

  MLIRTransforms
  LLVMCore
  LLVMSupport
)
