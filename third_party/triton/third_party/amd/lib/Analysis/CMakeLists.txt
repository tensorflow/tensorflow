add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
