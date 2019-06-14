set(LIBS
  MLIRAffineOps
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  LLVMSupport
)

add_llvm_library(MLIRAffineToGPU
  AffineToGPU.cpp
  AffineToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToGPU
)
add_dependencies(MLIRAffineToGPU ${LIBS})
target_link_libraries(MLIRAffineToGPU ${LIBS})
