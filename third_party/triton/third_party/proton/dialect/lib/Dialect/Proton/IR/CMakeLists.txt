add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonTableGen
  ProtonAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
