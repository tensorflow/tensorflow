add_llvm_library(MLIRGPUtoSPIRVTransforms
  GPUToSPIRV.cpp
  )

target_link_libraries(MLIRGPUtoSPIRVTransforms
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRStandardOps
  MLIRSPIRVConversion
  MLIRTransforms
  )
