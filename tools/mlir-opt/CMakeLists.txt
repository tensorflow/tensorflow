set(LIBS
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRLLVMIR
  MLIRParser
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRSupport
  MLIRVectorOps
)
add_executable(mlir-opt
  mlir-opt.cpp
)

whole_archive_link(mlir-opt ${LIBS})
target_link_libraries(mlir-opt MLIRIR ${LIBS} LLVMSupport)
