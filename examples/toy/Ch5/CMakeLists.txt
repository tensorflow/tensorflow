add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyCh5CombineIncGen)

add_toy_chapter(toyc-ch5
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/DeadFunctionEliminationPass.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch5 ToyCh5ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch5 ToyCh5OpsIncGen)
add_dependencies(toyc-ch5 ToyCh5CombineIncGen)
add_dependencies(toyc-ch5 MLIRCallOpInterfacesIncGen)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch5
  PRIVATE
    MLIRAffineOps
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRStandardOps
    MLIRTransforms)

whole_archive_link(toyc-ch5
  MLIRAffineOps
  MLIRStandardOps
  )
