set(LLVM_TARGET_DEFINITIONS CallInterfaces.td)
mlir_tablegen(CallInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CallInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCallOpInterfacesIncGen)
