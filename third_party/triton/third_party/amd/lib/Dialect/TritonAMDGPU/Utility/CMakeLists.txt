add_triton_library(TritonAMDUtils
  CommonUtils.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
