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

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name GmlSt)
add_public_tablegen_target(MLIRGmlStPassIncGen)

set(LLVM_TARGET_DEFINITIONS test_passes.td)
mlir_tablegen(test_passes.h.inc -gen-pass-decls -name GmlStTest)
add_public_tablegen_target(MLIRGmlStTestPassIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})


add_mlir_library(GmlStPasses
  collapse_materialize_ops/collapse_materialize_ops.cc
  collapse_shape/collapse_shape.cc
  fusion/fusion.cc
  gml_st_simtfy/gml_st_simtfy.cc
  gml_st_to_scf/gml_st_to_scf.cc
  gml_st_to_gpu/gml_st_to_gpu.cc
  greedy_tiling_and_fusion/greedy_tiling_and_fusion.cc
  lower_vector_contract/lower_vector_contract.cc
  peeling/peeling.cc
  tiling/tiling.cc
  tiling_cwise/tiling_cwise.cc
  tiling_gpu_warp/tiling_gpu_warp.cc
  tiling_softmax/tiling_softmax.cc
  transform_map_for_cpu/transform_map_for_cpu.cc
  transform_matmul_for_cpu/transform_matmul_for_cpu.cc
  transform_reduce_for_cpu/transform_reduce_for_cpu.cc
  transform_transpose_for_cpu/transform_transpose_for_cpu.cc
  transform_scatter_for_cpu/transform_scatter_for_cpu.cc
  vectorization/vectorization.cc

  DEPENDS
  MLIRGmlStPassIncGen
  MLIRGmlStUtils

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStTilingInterface
  GmlStTilingInterfaceImpl
  MLIRDestinationStyleOpInterface
  MhloDialect
  MLIRDialectUtils
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUOps
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCFUtils
  MLIRSupport
  MLIRVectorDialect
)

add_mlir_library(GmlStTransforms
  transforms.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStDialect
  MLIRAffineDialect
  MLIRDialectUtils
  MLIRIR
)

add_mlir_library(GmlStTestPasses
  test_passes.cc

  DEPENDS
  MLIRGmlStTestPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStBufferizableOpInterface
  GmlStDialect
  GmlStTransforms
  MLIRPass
  MLIRTransforms
)
