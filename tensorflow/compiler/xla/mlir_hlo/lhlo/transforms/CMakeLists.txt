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

set(LLVM_TARGET_DEFINITIONS lmhlo_passes.td)
mlir_tablegen(lmhlo_passes.h.inc -gen-pass-decls -name AllLmhlo)
add_public_tablegen_target(MLIRLmhloPassIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_library(LmhloPasses
  legalize_to_tensor_op/legalize_to_tensor_op.cc
  lhlo_elemental_utils.cc
  lhlo_legalize_to_affine/lhlo_legalize_to_affine.cc
  lhlo_legalize_to_gpu/lhlo_legalize_to_gpu.cc
  lhlo_legalize_to_parallel_loops/lhlo_legalize_to_parallel_loops.cc

  DEPENDS
  MLIRlhlo_opsIncGen
  MLIRLmhloPassIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LmhloDialect
  MLIRComplexDialect
  MLIRGPUOps
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMhloUtils
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
