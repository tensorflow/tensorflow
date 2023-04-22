add_mlir_library(MLIRHLOAnalysis
  userange_analysis.cc

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
)

add_mlir_library(MLIRHLOTestAnalysis
  test_userange_analysis.cc

  DEPENDS
  LMHLOTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRHLOAnalysis
  MLIRAnalysis
  MLIRPass
  )
