add_mlir_public_c_api_library(MLIRHLOCAPIDialects
  Attributes.cpp
  Dialects.cpp
  Types.cpp
  LINK_LIBS PUBLIC
  ChloDialect
  MhloDialect
)
