set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/ToyDialect.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )
include_directories(include/)
target_link_libraries(toyc-ch4
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms)

