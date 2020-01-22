add_llvm_library(LLVMMLIRTableGen
  Argument.cpp
  Attribute.cpp
  Constraint.cpp
  Dialect.cpp
  Format.cpp
  Operator.cpp
  OpInterfaces.cpp
  OpTrait.cpp
  Pattern.cpp
  Predicate.cpp
  Type.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/TableGen
  )
target_link_libraries(LLVMMLIRTableGen LLVMSupport LLVMTableGen)
