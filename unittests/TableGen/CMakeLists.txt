set(LLVM_TARGET_DEFINITIONS enums.td)
mlir_tablegen(EnumsGenTest.h.inc -gen-enum-decls)
mlir_tablegen(EnumsGenTest.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTableGenEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS structs.td)
mlir_tablegen(StructAttrGenTest.h.inc -gen-struct-attr-decls)
mlir_tablegen(StructAttrGenTest.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRTableGenStructAttrIncGen)

add_mlir_unittest(MLIRTableGenTests
  EnumsGenTest.cpp
  StructsGenTest.cpp
  FormatTest.cpp
)

add_dependencies(MLIRTableGenTests MLIRTableGenEnumsIncGen)
add_dependencies(MLIRTableGenTests MLIRTableGenStructAttrIncGen)

target_link_libraries(MLIRTableGenTests
  PRIVATE LLVMMLIRTableGen MLIRIR)
