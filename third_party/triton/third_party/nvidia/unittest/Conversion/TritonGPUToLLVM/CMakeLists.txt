add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS TritonGPUToLLVM TritonNVIDIAGPUToLLVM
)
