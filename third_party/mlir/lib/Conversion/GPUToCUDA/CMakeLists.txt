if(MLIR_CUDA_CONVERSIONS_ENABLED)
  llvm_map_components_to_libnames(nvptx "NVPTX")

  add_llvm_library(MLIRGPUtoCUDATransforms
    ConvertKernelFuncToCubin.cpp
    ConvertLaunchFuncToCudaCalls.cpp
  )
  target_link_libraries(MLIRGPUtoCUDATransforms
    MLIRGPU
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRTargetNVVMIR
    ${nvptx}
  )
endif()
