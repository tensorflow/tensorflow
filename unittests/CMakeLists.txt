add_custom_target(MLIRUnitTests)
set_target_properties(MLIRUnitTests PROPERTIES FOLDER "MLIR Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Dialect)
add_subdirectory(IR)
add_subdirectory(Pass)
add_subdirectory(TableGen)
