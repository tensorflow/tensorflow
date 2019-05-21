add_mlir_unittest(MLIRSDBMTests
  SDBMTest.cpp
)
target_link_libraries(MLIRSDBMTests
  PRIVATE
  MLIRSDBM
)
