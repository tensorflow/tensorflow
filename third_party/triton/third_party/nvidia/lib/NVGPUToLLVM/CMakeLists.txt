add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen
    NVGPUIR
)
