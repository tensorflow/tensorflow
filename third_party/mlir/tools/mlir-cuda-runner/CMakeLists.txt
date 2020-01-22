set(LLVM_OPTIONAL_SOURCES
  cuda-runtime-wrappers.cpp
  mlir-cuda-runner.cpp
  )

if(MLIR_CUDA_RUNNER_ENABLED)
  if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building the mlir cuda runner requires the NVPTX backend")
  endif()

  # Configure CUDA runner support. Using check_language first allows us to give
  # a custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building the mlir cuda runner requires a working CUDA install")
  endif()

  # We need the libcuda.so library.
  find_library(CUDA_RUNTIME_LIBRARY cuda)

  add_llvm_library(cuda-runtime-wrappers SHARED
    cuda-runtime-wrappers.cpp
  )
  target_include_directories(cuda-runtime-wrappers
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    LLVMSupport
  )
  target_link_libraries(cuda-runtime-wrappers
    LLVMSupport
    ${CUDA_RUNTIME_LIBRARY}
  )

  set(FULL_LINK_LIBS
    MLIRAffineOps
    MLIRLoopToStandard
    MLIRGPU
    MLIRGPUtoCUDATransforms
    MLIRGPUtoNVVMTransforms
    MLIRLLVMIR
    MLIRStandardOps
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    MLIRTranslation
  )
  set(LIBS
    MLIRIR
    MLIRParser
    MLIREDSC
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRJitRunner
    MLIRSupport
    LLVMCore
    LLVMSupport
    ${CUDA_RUNTIME_LIBRARY}
  )
  add_llvm_executable(mlir-cuda-runner
    mlir-cuda-runner.cpp
  )
  add_dependencies(mlir-cuda-runner cuda-runtime-wrappers)
  target_include_directories(mlir-cuda-runner
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  llvm_update_compile_flags(mlir-cuda-runner)
  whole_archive_link(mlir-cuda-runner ${FULL_LINK_LIBS})
  target_link_libraries(mlir-cuda-runner PRIVATE ${FULL_LINK_LIBS} ${LIBS})

endif()
