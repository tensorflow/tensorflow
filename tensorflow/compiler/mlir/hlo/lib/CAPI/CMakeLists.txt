add_mlir_public_c_api_library(MLIRHLOCAPIDialects
  Attributes.cpp
  Dialects.cpp
  Types.cpp
  Passes.cpp
  LINK_LIBS PUBLIC
  ChloDialect
  MhloDialect
)
target_link_libraries(MLIRHLOCAPIDialects PRIVATE AllMhLoPasses)
