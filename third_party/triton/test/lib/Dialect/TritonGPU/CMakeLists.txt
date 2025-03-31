add_mlir_library(TritonTestDialectTritonGPU
  TestTC05MMAPipeline.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
