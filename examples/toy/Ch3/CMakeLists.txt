set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch3
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/ToyDialect.cpp
  )
include_directories(include/)
target_link_libraries(toyc-ch3
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms)
