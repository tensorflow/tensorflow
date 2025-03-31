set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonCombineIncGen)

add_triton_library(TritonTransforms
  Combine.cpp
  LoopInvariantCodeMotion.cpp
  LoopUnroll.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
