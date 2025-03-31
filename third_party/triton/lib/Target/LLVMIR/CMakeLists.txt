add_triton_library(TritonLLVMIR
        LLVMDIScope.cpp
        LLVMIRBreakPhiStruct.cpp

        DEPENDS
        LLVMIRIncGen

        LINK_LIBS
        ${CMAKE_DL_LIBS}
        PUBLIC
        MLIRArithToLLVM
        MLIRBuiltinToLLVMIRTranslation
        MLIRIndexToLLVM
        MLIRIR
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRNVVMToLLVMIRTranslation
        MLIRROCDLToLLVMIRTranslation
        MLIRSCFToControlFlow
        MLIRSupport
        MLIRTargetLLVMIRExport
        TritonGPUToLLVM
        )

set_source_files_properties(
        LLVMIRTranslation.cpp
        PROPERTIES
        COMPILE_FLAGS "-D__BUILD_DIR__=\\\"${CMAKE_BINARY_DIR}\\\"")
