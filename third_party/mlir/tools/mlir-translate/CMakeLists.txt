set(LIBS
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRTargetLLVMIR
  MLIRTargetNVVMIR
  MLIRTargetROCDLIR
  MLIRTranslation
  MLIRSupport
)
add_llvm_executable(mlir-translate
  mlir-translate.cpp
)
llvm_update_compile_flags(mlir-translate)
whole_archive_link(mlir-translate ${LIBS})
target_link_libraries(mlir-translate PRIVATE MLIRIR MLIRTranslateClParser ${LIBS} LLVMSupport)
