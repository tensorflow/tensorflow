add_mlir_library(TritonAMDGPUTestAnalysis
  TestAMDRangeAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
