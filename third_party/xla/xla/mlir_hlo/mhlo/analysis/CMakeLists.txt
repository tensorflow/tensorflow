add_mlir_library(MhloAnalysis
  shape_component_analysis.cc

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  LmhloDialect
)
