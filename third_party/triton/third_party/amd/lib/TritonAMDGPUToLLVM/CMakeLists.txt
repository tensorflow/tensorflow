add_triton_library(TritonAMDGPUToLLVM
    AtomicRMWOpsEmitter.cpp
    BufferOpsEmitter.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandHelper.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandWMMA.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    OptimizeLDSUsage.cpp
    OptimizeLDSUtility.cpp
    SPMDOpToLLVM.cpp
    SchedInstructions.cpp
    UpcastMXFPToLLVM.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonAMDGPUIR
    TritonProtonToLLVM
)
