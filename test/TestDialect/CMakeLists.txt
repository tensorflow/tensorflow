set(LLVM_OPTIONAL_SOURCES
  TestDialect.cpp
  TestPatterns.cpp
)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
mlir_tablegen(TestOps.h.inc -gen-op-decls)
mlir_tablegen(TestOps.cpp.inc -gen-op-defs)
mlir_tablegen(TestPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRTestOpsIncGen)

add_llvm_library(MLIRTestDialect
  TestDialect.cpp
  TestPatterns.cpp
)
add_dependencies(MLIRTestDialect
  MLIRTestOpsIncGen
  LLVMSupport
)
target_link_libraries(MLIRTestDialect
  LLVMSupport
)

add_llvm_executable(mlir-test-opt
  ${MLIR_MAIN_SRC_DIR}/../tools/mlir-opt/mlir-opt.cpp
  TestDialect.cpp
  TestPatterns.cpp
)
llvm_update_compile_flags(mlir-test-opt)
whole_archive_link(mlir-test-opt
  MLIRStandardOps
)
target_link_libraries(mlir-test-opt
  PRIVATE
  MLIRMlirOptLib
  MLIROptMain
  MLIRTypeUtilities
  LLVMSupport
)
