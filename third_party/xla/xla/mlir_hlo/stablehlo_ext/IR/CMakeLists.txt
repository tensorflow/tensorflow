# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 The StableHLO Authors.
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

add_mlir_library(StablehloExtensionBase
  PARTIAL_SOURCES_INTENDED
  base.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(StablehloExtensionOps
  PARTIAL_SOURCES_INTENDED
  stablehlo_ops.cpp

  DEPENDS
  StablehloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloExtensionBase
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  StablehloOps
)

target_include_directories(StablehloExtensionOps INTERFACE
  $<BUILD_INTERFACE:${STABLEHLO_SOURCE_DIR}>
  $<BUILD_INTERFACE:${STABLEHLO_BINARY_DIR}>
)
