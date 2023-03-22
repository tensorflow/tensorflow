#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_library(MhloRngUtils
  mhlo_rng_utils.cc

  DEPENDS
  MLIRhlo_opsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LLVMSupport
  MhloDialect
  MLIRArithDialect
  MLIRIR
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRRewrite
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

add_mlir_library(MhloScatterUtils
  mhlo_scatter_gather_utils.cc

  DEPENDS
  MLIRhlo_opsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
)

add_mlir_library(MhloTypeConversion
  type_conversion.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRIR
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRTensorDialect
  StablehloOps
)

add_mlir_library(HloToLinalgUtils
  legalize_to_linalg_utils.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LmhloDialect
  MhloDialect
  MhloTypeConversion
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRIR
  MLIRLinalgUtils
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
