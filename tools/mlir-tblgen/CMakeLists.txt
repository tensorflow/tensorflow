set(LLVM_LINK_COMPONENTS
  MLIRTableGen
  Support
  )

add_tablegen(mlir-tblgen MLIR
  EnumsGen.cpp
  LLVMIRConversionGen.cpp
  mlir-tblgen.cpp
  OpDefinitionsGen.cpp
  OpDocGen.cpp
  ReferenceImplGen.cpp
  RewriterGen.cpp
  SPIRVSerializationGen.cpp
  )
set_target_properties(mlir-tblgen PROPERTIES FOLDER "Tablegenning")
