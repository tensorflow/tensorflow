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

add_mlir_library(DiscRalPasses
  ral_inject_execution_context.cc
  ral_lower_to_library_call.cc

  DEPENDS
  MLIRdisc_ral_opsIncGen
  MLIRDiscRalPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  DiscRalDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
)

add_mlir_library(DiscRalToLLVM
  ral_legalize_to_llvm.cc

  DEPENDS
  MLIRdisc_ral_opsIncGen
  MLIRDiscRalPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  DiscRalDialect
  MLIRArithmetic
  MLIRArithmeticToLLVM
  MLIRArithmeticTransforms
  MLIRGPUOps
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
)

add_library(AllDiscRalPasses INTERFACE)
target_link_libraries(AllDiscRalPasses INTERFACE
  DiscRalPasses
  DiscRalToLLVM
)
