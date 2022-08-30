# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set(LLVM_OPTIONAL_SOURCES
  bufferizable_op_interface_impl.cc
)

add_mlir_library(GmlStTilingInterface
  tiling_interface.cc

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport

  DEPENDS
  MLIRGmlStTilingInterfaceIncGen
)

add_mlir_library(GmlStFusionInterface
  fusion_interface.cc

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRIR
  MLIRSupport

  DEPENDS
  MLIRGmlStFusionInterfaceIncGen
)

add_mlir_library(GmlStTilingInterfaceImpl
  tiling_interface_impl.cc

  LINK_LIBS PUBLIC
  GmlStTilingInterface
  MLIRArithmeticDialect
  MLIRAffineDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRIR
  MLIRSupport
)

add_mlir_library(GmlStFusionInterfaceImpl
  fusion_interface_impl.cc

  LINK_LIBS PUBLIC
  GmlStFusionInterface
  MLIRArithmeticDialect
  MLIRIR
  MLIRSupport
)

add_mlir_library(GmlStComposeSetInterface
  compose_set_interface.cc

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRGmlStComposeSetInterfaceIncGen
)

add_mlir_library(GmlStBufferizableOpInterface
  bufferizable_op_interface_impl.cc

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRSupport
)

add_mlir_library(GmlStPasses
  bufferize_tiled_loop.cc
  collapse_materialize_ops.cc
  compose_set_ops.cc
  fusion.cc
  gml_st_to_scf.cc
  tiling.cc
  vectorization.cc

  DEPENDS
  MLIRGmlStPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStComposeSetInterface
  GmlStFusionInterface
  GmlStFusionInterfaceImpl
  GmlStTilingInterface
  GmlStTilingInterfaceImpl
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSupport
  MLIRVectorDialect
)

add_mlir_library(GmlStTransforms
  transforms.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSCFUtils
)

add_mlir_library(GmlStTestPasses
  test_passes.cc

  DEPENDS
  MLIRGmlStTestPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStBufferizableOpInterface
  GmlStTransforms
  MLIRPass
  MLIRTransforms
)
