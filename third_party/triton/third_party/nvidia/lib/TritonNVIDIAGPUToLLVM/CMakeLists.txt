add_triton_library(TritonNVIDIAGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    ConvertWarpSpecializeToLLVM.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/MMAv5.cpp
    DotOpToLLVM/WGMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    TMAToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorMemoryToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    Fp4ToFpOpToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNVIDIAGPUConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonProtonToLLVM
)
