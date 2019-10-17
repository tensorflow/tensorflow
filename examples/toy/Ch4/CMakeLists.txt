add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyCh4CombineIncGen)

add_toy_chapter(toyc-ch4
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/DeadFunctionEliminationPass.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch4 ToyCh4OpsIncGen)
add_dependencies(toyc-ch4 ToyCh4ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch4 ToyCh4CombineIncGen)
add_dependencies(toyc-ch4 MLIRCallOpInterfacesIncGen)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(toyc-ch4
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms)

