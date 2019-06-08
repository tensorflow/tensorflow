set(LLVM_TARGET_DEFINITIONS enums.td)
mlir_tablegen(EnumsGenTest.h.inc -gen-enum-decls)
mlir_tablegen(EnumsGenTest.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTableGenEnumsIncGen)

add_mlir_unittest(MLIRTableGenTests
  EnumsGenTest.cpp
  FormatTest.cpp
)

add_dependencies(MLIRTableGenTests MLIRTableGenEnumsIncGen)

target_link_libraries(MLIRTableGenTests
  PRIVATE LLVMMLIRTableGen)
