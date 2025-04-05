add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
