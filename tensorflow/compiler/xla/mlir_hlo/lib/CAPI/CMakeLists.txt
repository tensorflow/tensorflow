add_mlir_public_c_api_library(MLIRHLOCAPIDialects
  Attributes.cc
  Dialects.cc
  Types.cc
  Passes.cc
  LINK_LIBS PUBLIC
  ChloDialect
  MhloDialect
  # For AllMhLoPasses:
  ChloPasses
  MhloPasses
  MhloToLhloConversion
  MhloToArithmeticConversion
  MhloToMemrefConversion
  MhloToStandard
  MhloToLinalg
  MhloShapeOpsToStandard
)
