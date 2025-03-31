add_triton_library(TritonAMDGPUIR
  Dialect.cpp

  DEPENDS
  TritonAMDGPUTableGen
  TritonAMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
