set(LLVM_OPTIONAL_SOURCES
  cblas.cpp
  cblas_interface.cpp
  )

add_llvm_library(cblas SHARED cblas.cpp)
add_llvm_library(cblas_interface SHARED cblas_interface.cpp)
target_link_libraries(cblas_interface PRIVATE cblas)

