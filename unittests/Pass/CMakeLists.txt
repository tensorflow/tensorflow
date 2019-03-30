add_mlir_unittest(MLIRPassTests
  AnalysisManagerTest.cpp
)
target_link_libraries(MLIRPassTests
  PRIVATE
  MLIRPass)
