add_triton_library(NVGPUIR
  Dialect.cpp

  DEPENDS
  NVGPUTableGen
  NVGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)
