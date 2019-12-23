set(LLVM_TARGET_DEFINITIONS LoopLikeInterface.td)
mlir_tablegen(LoopLikeInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(LoopLikeInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRLoopLikeInterfaceIncGen)


