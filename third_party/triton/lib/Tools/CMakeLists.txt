add_triton_library(TritonTools
  LayoutUtils.cpp
  LinearLayout.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)
