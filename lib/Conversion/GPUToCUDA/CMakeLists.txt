# The CUDA conversions are only available if we have a working CUDA install.
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
# Enable the CUDA language so that CMake finds the headers and library for us.
enable_language(CUDA)
add_llvm_library(MLIRGPUtoCUDATransforms
  ConvertKernelFuncToCubin.cpp
  ConvertLaunchFuncToCudaCalls.cpp
  )
target_include_directories(MLIRGPUtoCUDATransforms
  PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
target_link_libraries(MLIRGPUtoCUDATransforms
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  ${CUDART_LIBRARY}
  )
endif()
