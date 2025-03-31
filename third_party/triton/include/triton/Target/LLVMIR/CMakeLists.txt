set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLVMIR)
add_public_tablegen_target(LLVMIRIncGen)
