add_llvm_executable(mlir-sdbm-api-test
  sdbm-api-test.cpp
)

llvm_update_compile_flags(mlir-sdbm-api-test)

target_link_libraries(mlir-sdbm-api-test
  PRIVATE
  MLIRIR
  MLIRSDBM
  LLVMCore
  LLVMSupport
)

target_include_directories(mlir-sdbm-api-test PRIVATE ..)

whole_archive_link(mlir-sdbm-api-test
  MLIRSDBM
)
