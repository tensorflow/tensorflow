include(AddMLIRPython)

################################################################################
# Generate dialect-specific bindings.
################################################################################

add_mlir_dialect_python_bindings(MLIRBindingsPythonMhloOps
  TD_FILE MhloOps.td
  DIALECT_NAME mhlo_dialect)
