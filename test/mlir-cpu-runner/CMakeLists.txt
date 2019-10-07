set(LLVM_OPTIONAL_SOURCES
  cblas.cpp
  cblas_interface.cpp
  mlir_runner_utils.cpp
  )

add_llvm_library(cblas SHARED cblas.cpp)
target_compile_definitions(cblas PRIVATE cblas_EXPORTS)

add_llvm_library(cblas_interface SHARED cblas_interface.cpp)
target_link_libraries(cblas_interface PRIVATE cblas)

add_llvm_library(mlir_runner_utils SHARED mlir_runner_utils.cpp)
target_compile_definitions(mlir_runner_utils PRIVATE mlir_runner_utils_EXPORTS)

