add_mlir_public_c_api_library(MLIRHLOCAPIDialects
  Dialects.cpp
  Types.cpp
  LINK_LIBS PUBLIC
  ChloDialect
  MhloDialect
)
