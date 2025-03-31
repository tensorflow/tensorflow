add_subdirectory(lib)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCe_DIR}/lit.cfg.py
)

set(TRITON_TEST_DEPENDS
  triton-opt
  triton-tensor-layout
  triton-llvm-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

add_lit_testsuite(check-triton-lit-tests "Running the triton regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${TRITON_TEST_DEPENDS}
  )

set_target_properties(check-triton-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRITON-LIT-TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_TEST_DEPENDS})
