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
  add_debug_info/add_debug_info.cc
  collapse_shape/collapse_shape.cc
  compose_extract_insert_slice/compose_extract_insert_slice.cc
  cpu_tiling/cpu_tiling_pipeline.cc
  cpu_tiling/fusion_planning_for_cpu.cc
  cpu_tiling/transform_dot_for_cpu.cc
  cpu_tiling/transform_map_for_cpu.cc
  cpu_tiling/transform_matmul_for_cpu.cc
  cpu_tiling/transform_reduce_for_cpu.cc
  cpu_tiling/transform_reverse_for_cpu.cc
  cpu_tiling/transform_scatter_for_cpu.cc
  cpu_tiling/transform_sort_for_cpu.cc
  cpu_tiling/transform_transpose_for_cpu.cc
  fusion/fusion.cc
  gml_st_to_scf/gml_st_to_scf.cc
  optimize_vector_transfer/optimize_vector_transfer.cc
  peeling/peeling.cc
  rewrite_vector_ops/rewrite_vector_contract.cc
  rewrite_vector_ops/rewrite_vector_multi_reduction.cc
  rewrite_vector_ops/rewrite_vector_transpose.cc
  scalarization/scalarization.cc
  tiling/tile_by_one.cc
  tiling/tiling.cc
  tiling_softmax/tiling_softmax.cc
  vectorization/vectorization.cc
  vectorization/vectorize_copy.cc
  vectorization/vectorize_for_cpu.cc

  DEPENDS
  MLIRGmlStPassIncGen
  MLIRGmlStUtils

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDestinationStyleOpInterface
  MhloDialect
  MLIRDialectUtils
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
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
