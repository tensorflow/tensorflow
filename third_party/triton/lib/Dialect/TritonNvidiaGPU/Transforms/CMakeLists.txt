add_triton_library(TritonNvidiaGPUTransforms
  FenceInsertion.cpp
  KeepAccInTMem.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
