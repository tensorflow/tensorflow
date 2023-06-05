add_mlir_library(MhloAnalysis
  shape_component_analysis.cc

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  LmhloDialect
)

add_mlir_library(MhloTestAnalysis
  test_shape_component_analysis.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRHLOAnalysis
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
)
