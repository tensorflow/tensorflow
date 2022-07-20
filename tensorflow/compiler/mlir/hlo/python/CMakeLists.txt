include(AddMLIRPython)

declare_mlir_python_sources(MLIRHLOPythonSources)
declare_mlir_python_sources(MLIRHLOPythonExtensions)

################################################################################
# Generate dialect-specific bindings.
################################################################################

declare_mlir_python_sources(MLIRHLOPythonSources.Dialects
  ADD_TO_PARENT MLIRHLOPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRHLOPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/MhloOps.td
  SOURCES dialects/mhlo.py
  DIALECT_NAME mhlo)

 declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MLIRHLOPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/ChloOps.td
  SOURCES dialects/chlo.py
  DIALECT_NAME chlo)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(MLIRHLOPythonExtensions.Main
  MODULE_NAME _mlirHlo
  ADD_TO_PARENT MLIRHLOPythonExtensions
  SOURCES
    MlirHloModule.cc
  EMBED_CAPI_LINK_LIBS
    MLIRHLOCAPIDialects
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
################################################################################

add_mlir_python_common_capi_library(MLIRHLOCAPI
  INSTALL_COMPONENT MLIRHLOPythonModules
  INSTALL_DESTINATION python_packages/mlir_hlo/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_HLO_BINARY_DIR}/python_packages/mlir_hlo/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRHLOPythonSources
    MLIRHLOPythonExtensions
)

add_mlir_python_modules(MLIRHLOPythonModules
  ROOT_PREFIX "${MLIR_HLO_BINARY_DIR}/python_packages/mlir_hlo/mlir"
  INSTALL_PREFIX "python_packages/mlir_hlo/mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRHLOPythonSources
    MLIRHLOPythonExtensions
  COMMON_CAPI_LINK_LIBS
    MLIRHLOCAPI
  )
