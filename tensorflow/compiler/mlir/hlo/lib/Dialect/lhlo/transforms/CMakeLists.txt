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

add_mlir_library(LmhloPasses
  fusion_utils.cc
  input_inline_fusion_pass.cc
  legalize_to_tensor_op.cc
  lhlo_elemental_utils.cc
  lhlo_fuse_linalg.cc
  lhlo_fusion.cc
  lhlo_fusion_inliner.cc
  lhlo_legalize_roots_to_loops.cc
  lhlo_legalize_to_affine.cc
  lhlo_legalize_to_gpu.cc
  lhlo_legalize_to_parallel_loops.cc

  DEPENDS
  MLIRlhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LmhloDialect
  MLIRComplex
  MLIRGPUOps
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMhloUtils
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
