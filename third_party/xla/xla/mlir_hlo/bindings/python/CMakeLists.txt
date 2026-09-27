#
# Copyright 2021 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
