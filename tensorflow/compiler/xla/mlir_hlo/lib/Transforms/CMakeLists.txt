#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

add_mlir_library(MLIRBufferTransforms
  buffer_packing.cc
  buffer_reuse.cc
  bufferize.cc
  bufferize_pass.cc
  copy_removal.cc
  detensorize_scf_ops.cc
  generic_host_to_llvm.cc
  propagate_static_shapes_to_kernel.cc
  lower_index_cast_pass.cc
  scalarization.cc
  symbolic_shape_optimization.cc
  shape_simplification.cc
  tile_loops_pass.cc
  collapse_parallel_loops_to_1d_pass.cc

  DEPENDS
  LMHLOTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ChloOps
  GmlStDialect
  MhloDialect
  MLIRGPUOps
  MLIRHLOAnalysis
  MLIRIR
  MLIRPass
  MLIRShapeDialect
  MLIRTransforms
)

add_mlir_library(MLIRHLOGPUTransforms
  alloc_to_arg_pass.cc
  gpu_kernel_lowering_passes.cc
  gpu_fusion_rewrite.cc
  hlo_to_gpu_pipeline.cc
  unbufferize_pass.cc

  DEPENDS
  LMHLOGPUTransformsPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStPasses
  MLIRArithmeticTransforms
  MLIRGPUOps
  MLIRHLOAnalysis
  MLIRIR
  MLIRPass
  MLIRShapeDialect
  MLIRSCFTransforms
  MLIRTransforms
)

add_mlir_library(MLIRHLOGmlStPipeline
  gml_st_pipeline.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  GmlStPasses
  MLIRLinalgTransforms
  MLIRPass
)
