add_llvm_library(MLIRParser
  Lexer.cpp
  Parser.cpp
  Token.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Parser
  )
add_dependencies(MLIRParser MLIRIR MLIRAnalysis)
target_link_libraries(MLIRParser MLIRIR MLIRAnalysis)
