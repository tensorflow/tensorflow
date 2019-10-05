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
  MLIRIR
  LLVMSupport
  MLIRTypeInferOpInterfaceIncGen
)
target_link_libraries(MLIRTestDialect
  MLIRDialect
  MLIRIR
  LLVMSupport
)
