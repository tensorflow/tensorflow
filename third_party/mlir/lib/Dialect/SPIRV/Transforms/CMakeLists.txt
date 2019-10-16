add_llvm_library(MLIRSPIRVTransforms
  DecorateSPIRVCompositeTypeLayoutPass.cpp
  )

target_link_libraries(MLIRSPIRVTransforms
  MLIRPass
  MLIRSPIRV
  )
